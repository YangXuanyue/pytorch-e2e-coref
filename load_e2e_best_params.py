from model_utils import *
from model import Model

e2e_best_params = {
	name: val
	for name, shape, val  in np.load('e2e_best_params.npy', allow_pickle=True)
}

param_names = list(
	map(
		lambda l: l.strip().split('\t'),
		open('param_names.txt').readlines()
	)
)

e2e_param_names = list(
	map(
		lambda l: l.strip().split('\t'),
		open('e2e_param_names.txt').readlines()
	)
)

model = Model()
old_state_dict = model.state_dict().copy()
model_params = {
	name: param
	for name, param in model.named_parameters()
}

for line, e2e_line in zip(param_names, e2e_param_names):
	name, shape, *opr_str = line
	e2e_name, e2e_shape = e2e_line

	if opr_str:
		# p(i, j, k)
		opr_str, = opr_str
		dims = map(int, opr_str[2:-1].split(', '))
		model_params[name].data.copy_(
			torch.as_tensor(e2e_best_params[e2e_name]).permute(*dims)
		)
	else:
		model_params[name].data.copy_(
			torch.as_tensor(e2e_best_params[e2e_name])
		)



torch.save(
	model.state_dict(),
	'best_params.sd'
)