import datasets as ds
import rbm, time
import matplotlib.pyplot as plt

minst = ds.load_minst(test=True)
epochs = 1
batch_size = 10

binaryRBM = rbm.BinaryRBM(size_visible_layer=minst.inputs, size_hidden_layer=568,batch_size=batch_size)
error_list = []
for epoch_index in xrange(epochs):
    #xrange(minst.num_obs/batch_size-1):
    print "Epoch: %d" % epoch_index
    start_total_t = time.clock()
    for batch_index in xrange(100):
        start_t = time.clock()
        low_bound = batch_index*batch_size
        up_bound = (batch_index+1)*batch_size
        error = binaryRBM.mini_batch_contrastive_divergence(minst.X[low_bound:up_bound])
        error_list.append(error)
        stop_t = time.clock()
        print "Single batch update (sec): %f" % (stop_t - start_t)
    stop_total_t = time.clock()
    print "Epoch (min): %f" % ((stop_total_t - start_total_t) / 60)

plt.plot(error_list)
plt.ylabel('Error')
plt.xlabel('Batch')
plt.show()
