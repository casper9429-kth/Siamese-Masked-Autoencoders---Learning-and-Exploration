import torch
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter object to write to TensorBoard
writer = SummaryWriter()

# Define the function to plot
def f(x):
    return 1/(x+1)+ 0.9**x

# Generate x values
x = torch.linspace(-10, 10, 100)

# Evaluate the function at each x value
y = f(x)

# Add the plot to TensorBoard
for x in range(100):
    for y in range(100):
        val = ((x*100) + y)*(1/100)
        writer.add_scalar("loss", y[val], val)
        # 1 s delay
        import time
        time.sleep(0.1)

# Close the SummaryWriter
writer.close()

