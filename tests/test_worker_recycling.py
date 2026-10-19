"""Exercise real spawn/replacement with the controller's rolling schedule."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
import os
import pickle
import tempfile
import time
import unittest


def checkpoint_block(index, directory):
    directory = Path(directory)
    (directory / f'start_{index}').touch()
    if index == 0:
        deadline = time.monotonic() + 20
        while not (directory / 'start_2').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Next block was not started while block zero remained active')
            time.sleep(.01)
    result = dict(block=index, pid=os.getpid(), value=index * index)
    partial = directory / f'block_{index}.partial'
    with partial.open('wb') as saved:
        pickle.dump(result, saved)
    partial.replace(directory / f'block_{index}.pkl')
    return result


class WorkerRecyclingTests(unittest.TestCase):
    def test_spawn_recycles_each_block_and_keeps_files_with_rolling_refill(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = iter(range(6))
            results = []
            with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn'),
                                     max_tasks_per_child=1) as executor:
                pending = {executor.submit(checkpoint_block, index, directory)
                           for index in islice(jobs, 2)}
                while pending:
                    done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    self.assertTrue(done, 'Worker replacement stalled')
                    for future in done:
                        results.append(future.result())
                        index = next(jobs, None)
                        if index is not None:
                            pending.add(executor.submit(checkpoint_block, index, directory))
            self.assertEqual(len({result['pid'] for result in results}), 6)
            self.assertEqual(sorted(result['block'] for result in results), list(range(6)))
            for result in results:
                with (Path(directory) / f"block_{result['block']}.pkl").open('rb') as saved:
                    self.assertEqual(pickle.load(saved), result)
                with self.assertRaises(ProcessLookupError):
                    os.kill(result['pid'], 0)


if __name__ == '__main__':
    unittest.main()
