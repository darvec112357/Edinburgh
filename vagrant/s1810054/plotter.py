import matplotlib.pyplot as plt

x=[i*30 for i in [0,1,2,3,4,5,6,7,8]]
y1=[53.17,68.92,108.91,127.59,124.94,135.76,91.77,59.86,46.81]
y2=[12.34,19.94,27.46,35.67,44.88,47.27,53.27,49.89,39.59]
y3=[3.95,6.2,9.67,12.35,15.45,13.94,16.64,17.82,19.68]

plt.plot(x, y1,color='blue',  linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=6)
plt.plot(x, y2,color='orange',  linewidth = 3,
         marker='s', markerfacecolor='orange', markersize=6)
plt.plot(x, y3,color='grey',  linewidth = 3,
         marker='^', markerfacecolor='grey', markersize=6)
plt.xlabel('Window Size')
plt.ylabel('Throughput(Kbps)')
plt.xticks(x,[2**i for i in [0,1,2,3,4,5,6,7,8]])
plt.legend(labels=['Delay(5ms)','Delay(25ms)','Delay(100ms)'])
plt.savefig('2-3.jpg')
