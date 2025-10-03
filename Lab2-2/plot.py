import thicket as th
import matplotlib.pyplot as plt

thicket_data = th.Thicket.from_caliperreader(['1024-2.cali', '1024-4.cali', '1024-8.cali', '1024-16.cali', '1024-32.cali', '1024-64.cali'])

df = thicket_data.dataframe

worker_receive_data = df[df['name'] == 'worker_receive']

ax.plot(worker_receive_data['num_procs'], 
        worker_receive_data['Min time/rank'], 
        label='Min', marker='o')
ax.plot(worker_receive_data['num_procs'], 
        worker_receive_data['Max time/rank'], 
        label='Max', marker='s')
ax.plot(worker_receive_data['num_procs'], 
        worker_receive_data['Avg time/rank'], 
        label='Avg', marker='^')

ax.set_xlabel('Number of Processes')
ax.set_ylabel('Time (seconds)')
ax.set_title('Worker Receive Time - Matrix Size 1024x1024')
ax.legend()
ax.grid(True)
plt.show()