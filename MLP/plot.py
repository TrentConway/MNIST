import matplotlib.pyplot as plt

BATCH_SIZE=32

def plot(data, pred, title=""):
    fig = plt.figure()
    plt.suptitle('Visualise {}'.format(title), fontsize=20)
    for i in range(BATCH_SIZE):
        plt.subplot(4,8,i+1)
        plt.title("val: {}".format(pred[i]), fontsize=8)
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.axis('off')
    plt.show()


