import h5py

names = ('train', 'dev')

elmo_caches = {
	name: h5py.File(f'data/{name}.elmo.cache.hdf5', 'r', swmr=True)[:]
	for name in names
}

while True:
	...