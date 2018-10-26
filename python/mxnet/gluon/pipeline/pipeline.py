from ..data.dataloader import DataLoader
from ..trainer import Trainer
from mxnet import init, autograd, gluon
import time

class Pipeline():
    def __init__(self, transformer=None, estimator=None):
        if transformer:
            self.transformer = transformer
        if estimator:
            self.estimator = estimator
        

    def compile(self, loss, optimizer, learning_rate):
        self.loss = loss
        self.metrics = lambda output, label: (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()
        self.estimator.initialize(init=init.Xavier())
        self.trainer = Trainer(self.estimator.collect_params(), optimizer, {'learning_rate': learning_rate})
        self.transformer.hybridize()
        self.estimator.hybridize()
        

    def fit(self, dataset, epochs, batch_size):
        # create data loader
        dataset = dataset.transform_first(self.transformer)
        train_data = DataLoader(
           dataset , batch_size=batch_size, shuffle=True, num_workers=4)
        for epoch in range(epochs):
            train_loss, train_acc = 0., 0.
            tic = time.time()
            for data, label in train_data:
                # forward + backward
                with autograd.record():
                    output = self.estimator(data)
                    loss = self.loss(output, label)
                loss.backward()
                # update parameters
                self.trainer.step(batch_size)
                # calculate training metrics
                train_loss += loss.mean().asscalar()
                train_acc += self.metrics(output, label)
            print('Epoch {}: Loss: {}, Train acc {},Time {} sec'
                    .format(epoch, train_loss / len(train_data),
                        train_acc / len(train_data), time.time() - tic))
        
    def predict(self, dataset):
        preds = []
        X, y = dataset
        for data in X:
            data = self.transformer(data).expand_dims(axis=0)
            pred = self.estimator(data).argmax(axis=1)
            preds.append(pred.astype('int32').asscalar())
        return preds
            
    def validate(self, dataset, batch_size):
        valid_data = DataLoader(
            dataset.transform_first(self.transformer),
                batch_size=batch_size, num_workers=4)
        valid_acc = 0.
        for data, label in valid_data:
            valid_acc += self.metrics(self.estimator(data), label)
        print('totoal loss {}'.format(valid_acc / len(valid_data)))

    def save(self, path, epoch):
        self.estimator.export(path, epoch=epoch)
        self.transformer.export('transform', epoch=epoch)

    def load(self, path=None):
        self.transformer = gluon.nn.SymbolBlock.imports('transform-symbol.json', ['data'])
        self.estimator = gluon.nn.SymbolBlock.imports('net.params-symbol.json', ['data'], 'net.params-0000.params')


