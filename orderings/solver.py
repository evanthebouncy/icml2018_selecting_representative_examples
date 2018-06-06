from z3 import *
from gen import L
import random

class OrderSolver(object):

    def __init__(self):
        self.reset_solver()

    def reset_solver(self):
        self.solver = Solver()

        self.ordering = [Int('ans_{}'.format(i)) for i in xrange(L)]
        
        for o in self.ordering:
            self.solver.add(And(0 <= o, o <= L-1))

        self.solver.add(Distinct(self.ordering))

    def add_example(self, example):
        nums, truth = example
        num1, num2 = nums
        if truth:
            self.solver.add(self.ordering[num1] < self.ordering[num2])
        else:
            self.solver.add(Not(self.ordering[num1] < self.ordering[num2]))

    def solve(self):
        if self.solver.check() == sat:
            model = self.solver.model()
            order = []
            for i, o in enumerate(self.ordering):
                order.append((i, model[o].as_long()))
            order = sorted(order, key=lambda x: x[1])
            return map(lambda x: x[0], order)
        else:
            print unsat
            return None

    def add_temp(self, example):
        '''temporarily adds an example and checks if sat'''
        self.solver.push()
        nums, truth = example
        num1, num2 = nums
        if truth:
            self.solver.add(self.ordering[num1] < self.ordering[num2])
        else:
            self.solver.add(Not(self.ordering[num1] < self.ordering[num2]))
        is_sat = self.solver.check() == sat
        self.solver.pop()
        return is_sat

    def check_ambiguous(self, example):
        '''returns true if example is ambiguous'''
        nums, truth = example
        example2 = (nums, not truth)
        # is_sat1 = self.add_temp(example)
        is_sat2 = self.add_temp(example2)
        return is_sat2


def check_solved(soln, examples):
    incorrect = []
    for example in examples:
        nums, truth = example
        num1, num2 = nums

        ind1 = soln.index(num1)
        ind2 = soln.index(num2)

        if (ind1 < ind2) != truth:
            incorrect.append(example)
    return incorrect


class OrderCEGIS(object):

    def __init__(self):
        self.solver = OrderSolver()

    def _reset(self):
        self.solver.reset_solver()

    def solve(self, examples):
        # example = random.choice(examples)
        example = examples[0]
        used_examples = [example]

        while True:
            self.solver.add_example(example)
            synth = self.solver.solve()
            incorrect = check_solved(synth, examples)

            if len(incorrect) == 0:
                return used_examples
            else:
                # example = random.choice(incorrect)
                example = incorrect[0]
                used_examples.append(example)


if __name__ == '__main__':
    # check if solver works
    from gen import *

    # s = OrderSolver()

    # for _ in xrange(100):
    #     truth = gen_ordering()
    #     examples = get_data(truth)
    #     for e in examples:
    #         s.add_example(e)
    #     synth = s.solve()
    #     print 'truth:', truth
    #     print 'synth:', synth
    #     assert synth
    #     s.reset_solver()

    # # reset solver and check an "ambiguous" example
    # print 'Trying ambiguous example...'
    # example = ((5, 1), False)
    # print 'is ambiguous:', s.check_ambiguous(example)
    # print 'Checking non-ambiguous example...'
    # s.add_example(example)
    # print 'is ambiguous:', s.check_ambiguous(example)

    truth = gen_ordering()
    examples = get_data(truth)
    c = OrderCEGIS()
    print truth
    print examples
    print c.solve(examples)
