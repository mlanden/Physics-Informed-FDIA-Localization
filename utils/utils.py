import json
import torch
import torch.multiprocessing as mp
from typing import List
import queue
import traceback
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt



def to_complex(s: str):
    if len(s) == 0:
        return complex(0)
    else:
        s = s.replace(" ", "")
        s = s.replace("(", "")
        s = s.replace(")", "")
        if "j" in s:
            s = s.replace("j", "") + "j"
        s = s.replace("i", "j")
        return complex(s)
        

def make_roc_curve(eval_file):
    with open(eval_file, "r") as fd:
        data = json.load(fd)

    labels = []
    scores = []
    for score, label in data:
        scores.append(score)
        labels.append(label)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("mean invarants.png")


def evaluate_loss(loss_objects: List, states: torch.Tensor, outputs: List[List[torch.Tensor]], targets: torch.Tensor,
                  n_workers) -> torch.Tensor:
    if n_workers <= 0:
        raise RuntimeError("Cannot use multiprocessing for 0 worker")

    tasks = mp.JoinableQueue()
    results = mp.JoinableQueue()
    work_completed_events = [mp.Event() for _ in range(n_workers)]
    stop_event = mp.Event()
    for i in range(len(loss_objects)):
        tasks.put(i)
    print(f"Number of tasks: {tasks.qsize()}", flush=True)
    n_tasks = tasks.qsize()

    workers = [mp.Process(target=_invariant_worker, args=(i, loss_objects, states, outputs, targets, tasks, results,
                                                          work_completed_events, stop_event)) for i in range(n_workers)]
    for worker in workers:
        worker.start()

    count = 0
    losses = torch.zeros((len(loss_objects), states.shape[0]))
    while count < n_tasks:
        try:
            id_, confidence = results.get(timeout=0.1)
            losses[id_, :] = confidence
            results.task_done()
            count += 1
            print("\r", end="", flush=True)
            print(f"{count} / {len(loss_objects)} invariants completed", end="", flush=True)
        except queue.Empty:
            pass
        except Exception as e:
            print("Main", e)
    stop_event.set()
    tasks.join()
    results.join()
    for worker in workers:
        worker.join()
    losses = torch.t(losses)
    print("\nInvariants evaluated")
    return losses


def _invariant_worker(rank: int, loss_objects: List, states: torch.Tensor, outputs: List[torch.Tensor], targets: torch.Tensor,
                      tasks: mp.JoinableQueue, results: mp.JoinableQueue, worker_end_events: List[mp.Event],
                      stop_event: mp.Event):
    while tasks.qsize() > 0:
        try:
            inv_id = tasks.get(timeout=0.1)
            invariant = loss_objects[inv_id]
            confidence = invariant.confidence_loss(states, outputs, targets)
            results.put((inv_id, confidence))
            tasks.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            traceback.print_exc()

    worker_end_events[rank].set()
    stop_event.wait()
