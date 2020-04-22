import numpy as np
from train import train

from hyperopt import hp, fmin, tpe, space_eval, Trials
from hyperopt.mongoexp import MongoTrials

trials = Trials()
# trials = MongoTrials('mongo://localhost:1234/jigsaw_db/jobs', exp_key='r002')

# define a search space
space = dict(
    optimizer='LAMB', #hp.choice('optimizer', ['LAMB', 'AdamW']),
    lr=hp.choice('lr', np.logspace(-6, -4, num=99)),
    weight_decay=hp.choice('weight_decay', np.logspace(-7, -4, num=99)),
#     loss_fn='focal',
    label_smoothing=hp.choice('label_smoothing', np.linspace(0.001, 0.05, num=99)),
#     pos_weight=hp.choice('pos_weight', np.linspace(1, 6, num=99)),
#     gamma=hp.choice('gamma', np.linspace(0.0, 3.0, num=99)),
    epochs=30,
    stages=10,
    dataset='../input/jigsaw-mltc-ds/jigsaw_mltc_ds436001s.npz',
    gcs='hm-eu-w4',
    path=f'jigsaw/h000',
    tpu_id='t8a',
    seed=3,
)

# minimize the objective over the space
def objective(args):
#     print('### ASD ### !!! ### $$$ %%%% &&&&')
    return -train(**args)

best = fmin(objective, space, trials=trials, algo=tpe.suggest, max_evals=6)

print(space_eval(space, best))
