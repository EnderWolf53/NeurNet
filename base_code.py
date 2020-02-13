import numpy as np
import pandas as pd
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from scipy import stats
from tensorflow.keras.utils import plot_model
import sklearn.preprocessing
import pickle


class CategoricalEncoder():
    def __init__(self):
        self._classes = {}
        self.size = 0

    def fit(self, X):
        uniques = np.unique(X)
        for u in uniques:
            if not u in self._classes:
                self._classes[u] = self.size
                self.size += 1

    def transform(self, X, unknown_category=True):
        result = np.full(X.shape, self.size, dtype=np.int32)
        for i in range(len(X)):
            try:
                result[i] = self._classes[X[i]]
            except KeyError:
                if unknown_category:
                    result[i] = self.size
                else:
                    raise KeyError
        return result

    def fit_transform(self, X, unknown_category=True):
        self.fit(X)
        return self.transform(X, unknown_category=unknown_category)

class ReproductionErrorLayer(tf.keras.layers.Layer):
    def __init__(self, loss, **kwargs):
        self.output_dim = 1
        self.loss = loss
        super(ReproductionErrorLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        super(ReproductionErrorLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        result = None
        assert isinstance(x, list)

        result = self.loss(x[0], x[1])

        return tf.reshape(result, shape=[-1, 1])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        return (input_shape[0][0], self.output_dim)

    def get_config(self):
        config = {
            'loss': self.loss
        }
        base_config = super(ReproductionErrorLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def label_to_int(label):
    if label =="Normal":
        return 0
    return 1

# This is the list of variables that will be used by the autoencoder
cols = ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
"rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
"synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
"ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
"ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
"ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"]

# TODO 1: Identify variables' types
def load_data(path, train=False):
    global cols
    df = pd.read_csv(path)[cols + ["attack_cat"]]
    inp = dict()
    out = dict()
    labels = None
    for key in df.columns:
        if key in ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "synack", "ackdat",
        "smean", "dmean", "response_body_len", "ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst"]:
            inp[key] = df[key].values.astype(np.float32)
            # TODO 2: transform variables
        # Caegorical varaiables
        elif key in ["proto", "service", "state"]:
            if train==True:
                inter = CategoricalEncoder()
                inp[key] = inter.fit_transform(df[key])
                pickle.dump(inter, open(key,'wb'))
            else:
                inter = CategoricalEncoder()
                pickle.load(open(key, 'rb'))
                inp[key] = inter.fit_transform(df[key])

            print(inp[key])
            continue

        elif "attack_cat" in key:
            labels = df[key].map(label_to_int).values
            continue

        # Label is hidden for the unknown.csv dataset
        elif "hidden_label" in key:
            labels = df[key].values
            continue

        else:
            inp[key] = df[key].values.astype(bool)

        if train:
            out[key+'-output'] = inp[key]
    if train:
        return inp, out, labels
    else:
        return inp, labels



def create_training_model(variables):
    inputs = []
    tensors = []

    # TODO 3.1: Define the encoding part specific to each input type
    for key in variables:
        inp = None
        x = None

        # Binary values
        if key in ["TODO 1"]:
            inp = None
            x = None

        # Categorical
        elif key in ["TODO 1"]:
            inp = None
            x = None

        # Numeric
        else:
            inp = None
            x = None

        inputs.append(inp)
        tensors.append(x)

    # Regroup all the inputs
    encoder = tf.keras.layers.Concatenate()(tensors)

    # TODO 3.2: Define the central part of the autoecoder
    decoder = None

    losses = {}
    outputs = []

    # TODO 3.3: Define the decoding part and loss specific to each input type
    for key in variables:
        loss = None
        x = None
        # Binary values
        if key in ["TODO 1"]:
            loss = None
            x = None

        # Categorical
        elif key in ["TODO 1"]:
            loss = None
            x = None

        # Numeric
        else:
            loss = None
            x = None

        losses[key+"-output"].append(loss)
        outputs.append(x)

    return tf.keras.Model(inputs, outputs), losses

def create_inference_model(trained_model, losses, data):
    # Integrate loss functions directly into the model
    loss_outs = []
    for key in losses:
        in_name = key.replace("-output", "")
        layer = ReproductionErrorLayer(losses[key])([trained_model.get_layer(in_name).output, trained_model.get_layer(key).output])
        loss_outs.append(layer)

    # Build temporary model to calibrate each loss
    tmp = tf.keras.Model(trained_model.inputs, loss_outs)
    error = tmp.predict(data, batch_size=1024)
    scalers = []
    for i in range(len(error)):
        # TODO 4: Compute parameters useful for calibration
        params = None
        scalers.append(tf.keras.layers.Lambda(loss_scaler(params))(tmp.outputs[i]))


    return tf.keras.Model(tmp.inputs, tf.keras.layers.Add()(scalers))

def loss_scaler(params):
    def fn(x):
        # TODO 4: scaling function
        # Use tensorflow supported functions and operators only
        return x
    return fn


def train_model(model, losses, data):
    model.compile(loss=losses, optimizer='adam')
    plot_model(model, to_file='autoencoder.png', show_shapes=True)
    x, y, _ = data
    model.fit(x, y, verbose=2, batch_size=1024, epochs=1000, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, min_delta=0.0001, restore_best_weights=True)])

    inf_model = create_inference_model(model, losses, x)

    return inf_model

def find_threshold(normal_scores, anormal_scores):
    # TODO 5: Finding threshold
    return 0


train_data = load_data("train.csv", train=True)
model, losses = create_training_model([k for k in train_data[0]])
model = train_model(model, losses, train_data)

test_data, y, labels = load_data("evaluate.csv")
scores = model.predict(test_data, batch_size=4096)

normal_ids = np.where(labels == 0)
anormal_ids = np.where(labels == 1)

threshold = find_threshold(scores[normal_ids], scores[anormal_ids])

# TODO 6: analyze "unknown.csv"
