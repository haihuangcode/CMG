from gensim.models import KeyedVectors


class ModelContainer(object):
    def __init__(self):
        super(ModelContainer, self).__init__()
        self.model = {}
        self.data = {}

    def save_model(self, model, name, data, criterion, greater_better=True):
        assert criterion in data, "Incompatible criterion name!"
        if (name not in self.model) or ((self.data[name][criterion] <= data[criterion]) == greater_better):
            self.model.update({name: model})
            self.data.update({name: data})
        return self.data[name]

    def fetch_model(self, name):
        assert name in self.model, "Invalid model name!"
        return self.model[name], self.data[name]


def merge_dicts(a, b):
    for key, value in b.items():
        a[key] = value + a[key] if key in a else value
    return a


class ValueContainer(object):
    def __init__(self):
        super().__init__()
        self.data = {}

    def reset(self, model_name):
        self.data[model_name] = {"count": 0}

    def update(self, model_name, metrics, step=1):
        if model_name not in self.data:
            self.reset(model_name)
        if not isinstance(metrics, dict):
            metrics = {"__default__": metrics}
        self.data[model_name] = merge_dicts(self.data[model_name], metrics)
        self.data[model_name]["count"] += step

    def calculate_average(self, model_name, reset=True):
        result = {}
        cum_result = self.data[model_name]
        for key in cum_result:
            if key != "count":
                result[key] = cum_result[key] / cum_result["count"]
        if reset:
            self.reset(model_name)
        if "__default__" in result:
            return result["__default__"]
        return result


class ResourceContainer:
    def __init__(self):
        super().__init__()
        self.vocab = None
        self.resource = {}

    def save_resource(self, name, resource):
        self.resource[name] = resource

    def fetch_resource(self, name):
        if name in self.resource:
            return self.resource[name]
        return None

    def fetch_vocab(self, vocab_path):
        if self.vocab is None:
            self.vocab = KeyedVectors.load_word2vec_format(vocab_path, binary=True)
        return self.vocab



modelContainer = ModelContainer()
metricsContainer = ValueContainer()
resourceContainer = ResourceContainer()
