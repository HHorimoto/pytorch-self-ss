import matplotlib.pyplot as plt

def plot(train, test, metric):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.plot(train, label='train')
    plt.plot(test, label='test')
    plt.legend()
    plt.savefig(metric+'.png')