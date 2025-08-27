import abc


class Pipeline(abc.ABC):
    @abc.abstractmethod
    def run(self, ctx):
        pass