from evaluator import Evaluator
from variation import VariationOperator
from signal_slot import EventLoopObject, signal, Timer


class Trainer(EventLoopObject):
    def __init__(self, mutator, evaluator, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.mutator: VariationOperator = mutator
        self.evaluator: Evaluator = evaluator
        self.train_timer: Timer = None

    @signal
    def stop(self): pass

    def on_stop(self):
        self.train_timer.stop()

    def periodic(self, period, callback):
        return Timer(self.event_loop, period).timeout.connect(callback)

    def on_start(self):
        self.train_timer = self.periodic(1.0, self.train)

    def train(self):
        mutated_policies = self.mutator.maybe_mutate_new_batch()
        if mutated_policies is not None:
            self.evaluator.evaluate_batch(mutated_policies)

    def get_components(self):
        return self.mutator, self.evaluator