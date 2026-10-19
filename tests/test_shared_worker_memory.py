"""Actual spawn checks for initialized RAM inputs, round updates and cleanup."""
import ast
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.util import Finalize
from pathlib import Path
import os
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def start_worker(inputs, descriptors, events, gate, closed):
    global scope, start_gate
    start_gate = gate
    scope = dict(np=np, SharedMemory=SharedMemory)
    def register(obj, callback, **settings):
        def close():
            callback()
            closed.send(('closed', os.getpid(), callback.__self__.name))
        return Finalize(obj, close, **settings)
    scope['Finalize'] = register
    tree = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'initialize_worker')
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / 'eric_subtraction_all.py'), 'exec'), scope)
    scope['initialize_worker'](inputs, descriptors)
    events.put(('initialized', os.getpid()))


def observe_round(block, template_table, wait_gate=False):
    if wait_gate:
        assert start_gate.wait(timeout=20)
    time.sleep(.02)
    return dict(pid=os.getpid(), block=block,
        observation=scope['lisa_data_fd'].copy(), covariance=scope['effective_noise_covariance'].copy(),
        template_table=template_table.copy())


class SharedWorkerMemoryTests(unittest.TestCase):
    def test_three_spawned_workers_initialize_once_and_see_next_round(self):
        ctx = get_context('spawn')
        events, gate = ctx.Queue(), ctx.Event()
        closed_reader, closed_writer = ctx.Pipe(duplex=False)
        table = np.arange(33, dtype=float).reshape(3, 11)
        inputs = (np.array([[.003, .00301]]), 1.e-5, {'f': 2}, 2048., .1, 16.,
                  np.array([[-1.e-10, -1.e-25], [1.e-25, 1.e-10]]), (32, 64), 64,
                  np.array([0.]), {'nested': [np.zeros((2, 3))]})
        descriptors, parent = {}, {}
        try:
            with ExitStack() as resources:
                for name, shape in [('lisa_data_fd', (4, 8)), ('effective_noise_covariance', (8, 3, 3))]:
                    memory = SharedMemory(create=True, size=int(np.prod(shape)) * 16)
                    resources.callback(memory.unlink)
                    resources.callback(memory.close)
                    parent[name] = np.ndarray(shape, dtype=complex, buffer=memory.buf)
                    resources.callback(parent.pop, name)
                    parent[name][:] = 1 if name == 'lisa_data_fd' else 2
                    descriptors[name] = (memory.name, shape, np.dtype(complex))
                with ProcessPoolExecutor(max_workers=3, mp_context=ctx,
                        initializer=start_worker, initargs=(inputs, descriptors, events, gate, closed_writer)) as executor:
                    # Block all three first tasks until all workers have initialized.
                    futures = [executor.submit(observe_round, i, table, True) for i in range(3)]
                    initialized = [events.get(timeout=20) for _ in range(3)]
                    self.assertTrue(all(kind == 'initialized' for kind, _ in initialized))
                    pids = {pid for _, pid in initialized}
                    self.assertEqual(len(pids), 3)
                    gate.set()
                    first = [future.result(timeout=20) for future in futures]
                    for result in first:
                        np.testing.assert_array_equal(result['observation'], parent['lisa_data_fd'])
                        np.testing.assert_array_equal(result['covariance'], parent['effective_noise_covariance'])
                        np.testing.assert_array_equal(result['template_table'], table)
                    # Reuse the same pool and buffers after the completed first round.
                    parent['lisa_data_fd'][1:] = 3
                    parent['effective_noise_covariance'][:] = 4
                    second = list(executor.map(observe_round, range(6), [table[1:]] * 6, chunksize=1))
                    for result in second:
                        self.assertIn(result['pid'], pids)
                        np.testing.assert_array_equal(result['observation'], parent['lisa_data_fd'])
                        np.testing.assert_array_equal(result['covariance'], parent['effective_noise_covariance'])
                        np.testing.assert_array_equal(result['template_table'], table[1:])
                    np.testing.assert_array_equal(parent['lisa_data_fd'][0], 1)
                closed = []
                for _ in range(2 * len(pids)):
                    self.assertTrue(closed_reader.poll(20), 'worker handles did not close')
                    closed.append(closed_reader.recv())
                self.assertEqual(sorted((pid, name) for kind, pid, name in closed if kind == 'closed'),
                    sorted((pid, name) for pid in pids for name, _, _ in descriptors.values()))
            self.assertFalse(parent)
            for name, _, _ in descriptors.values():
                with self.assertRaises(FileNotFoundError):
                    SharedMemory(name=name)
        finally:
            events.close()
            events.join_thread()
            closed_reader.close()
            closed_writer.close()


if __name__ == '__main__':
    unittest.main()
