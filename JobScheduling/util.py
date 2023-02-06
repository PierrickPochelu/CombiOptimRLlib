from typing import *
import numpy as np


class Item:
    def __init__(self, start, duration, weight, identifier, submitted_date):
        self.start = start
        self.duration = duration
        self.weight = weight
        self.identifier = identifier
        self.submitted_date = submitted_date

    def __str__(self):
        return f"id:{self.identifier} duration:{self.duration}, weigh:{self.weight}, " \
               f"start_date:{self.start}, submitted_date:{self.submitted_date}"


class SchedulingSim:
    def __init__(self, capacity):
        self.capacity = capacity  # constant
        self.items = []  # list of tuple (start, duration, weight, id)
        self.timeline_ressource = []  # ressource consumption at time t

    def add_item(self, I: Item) -> Optional[int]:
        """
        Item info as input, produces number of new time steps and if resource overflow occurs
        :param start:
        :param duration:
        :param weight:
        :param id:
        :return:
        """
        start, duration, weight, identifier = I.start, I.duration, I.weight, I.identifier

        self.items.append((start, duration, weight, identifier))
        increase_required_time = 0
        for t in range(start, start + duration, 1):
            while len(self.timeline_ressource) <= t:
                # we need one line more line
                increase_required_time += 1
                self.timeline_ressource.append(0)

            if self.can_fit(t, weight):
                self.timeline_ressource[t] += weight
            else:
                # crash
                return None

        return increase_required_time

    def can_fit(self, t, weight_item):
        current_conso = self.timeline_ressource[t] if t < len(self.timeline_ressource) else 0.
        return current_conso + weight_item < self.capacity

    def get_table(self):
        """ for debugging and usefull representation for measuring some metrics """
        matrix = np.zeros((len(self.timeline_ressource), self.capacity), dtype=int)

        for I in self.items:
            start, duration, weight, identifier = I.start, I.duration, I.weight, I.identifier
            for i in range(start, start + duration):
                remaining_w = weight
                j = 0
                while remaining_w > 0 and j < self.capacity:
                    if matrix[i, j] == 0:  # if out of bound, we overflow the capacity
                        matrix[i, j] = identifier
                        remaining_w -= 1
                    j += 1
                if remaining_w != 0:
                    print(f"Alert line {i} item {identifier} is not well stored")
        return matrix

    def get_items_at(self, t):
        items = []
        for I in self.items:
            start, duration, weight, identifier = I.start, I.duration, I.weight, I.identifier
            if start <= t and t < start + duration:
                items.append(I)
        return items

    def evaluate(self, horizon):
        # ressource util
        mean_util = np.mean(self.timeline_ressource[:horizon])

        # Average waiting
        wait = []
        for item in self.items:
            wait_i = item.start - item.submitted_date
            wait.append(wait_i)
        wait = np.array(wait)

        max_wait = np.max(wait)
        average_wait = np.mean(wait)
        rate_wait = np.mean(wait != 0)
        return {"mean_util": mean_util,  # to maximize
                "max_wait": max_wait,  # to minimize
                "average_wait": average_wait,  # to minimize
                "rate_wait": rate_wait  # to minimize
                }


def cond_latest(A, B):
    return A.submitted_date <= B.submitted_date


def cond_biggest(A, B):
    return A.weight >= B.weight


def cond_smallest(A, B):
    return A.weight <= B.weight


def cond_shortest(A, B):
    return A.duration <= B.duration


def cond_longest(A, B):
    return A.duration >= B.duration


def get_fitting_item_according_cond(items: List[Item], avail_ressources: int, cond: Callable):
    """biggest fitting items"""
    if len(items) == 0:
        return None

    best_and_latest_item = None
    for I in items:
        if I.weight <= avail_ressources:
            if best_and_latest_item is None or cond(I, best_and_latest_item):
                best_and_latest_item = I
    return best_and_latest_item

def starvation_penalty(items:List[Item], t:int, A:float, B:float, MAX:float)->float:
    """ polynomial penalty. Large A may reduce the number of iems added to the pending_queue. Large B reduces starvation. """
    p=0
    for I in items:
        w=I.submitted_date-t
        p+=A*w+B*(w**2)

    return -min(p, MAX)