from runner import Runner
from model_utils import *

runner = Runner()
# runner.load_ckpt()
# trainer.validate(name='valid', saves_results=True)
runner.model.load_state_dict(
	torch.load('best_params.sd').cuda()
)
runner.evaluate()