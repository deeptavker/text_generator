import numpy as np

fname_i = '../raw_io_files/input_train_'
fname_o = '../raw_io_files/input_target_'

tname = raw_input()
file_counter = np.load('../var_files/batch_counter.npy')[0]


batch_size = 128

i = np.load(fname_i + tname + '.npy')
o = np.load(fname_o + tname + '.npy')

no_of_samples = i.shape[0]
print(no_of_samples)

nbatch = no_of_samples // batch_size

for k in range(nbatch):
    new_i = i[k * batch_size : (k + 1) * batch_size]
    new_o = o[k * batch_size : (k + 1) * batch_size]
    np.save('../io_files/io_128_i_' + str(file_counter), new_i)
    np.save('../io_files/io_128_o_' + str(file_counter), new_o)
    file_counter += 1

new_i = i[nbatch * batch_size : ]
new_o = o[nbatch * batch_size : ]

if new_i.shape[0] != 0:
    np.save('../io_files/io_128_i_' + str(file_counter), new_i)
    np.save('../io_files/io_128_o_' + str(file_counter), new_o)
    file_counter += 1
    
file_counter = np.array([file_counter])
np.save('../var_files/batch_counter.npy', file_counter)