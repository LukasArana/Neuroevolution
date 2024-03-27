import matplotlib.pyplot as plt
env = "InvertedPendulum"
with open(f"results/random/{env}.txt", "r") as f:
    a = list(map(float, f.readlines()[0].split(",")[:-1]))
    print(len(a)) 
a = a[:1000]
#plt.scatter(x = list(range(len(a))), y = a)
plt.plot(a)
plt.xlabel('Seed')
plt.ylabel('Pendulum Reward')
plt.savefig(f'results/random/figures/{env}.png')
