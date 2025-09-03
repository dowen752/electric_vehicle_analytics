import matplotlib.pyplot as plt
import pandas as pd
import os


# numbersX = [1, 2, 3, 4, 5]
# numbersY = [10, 20, 25, 30, 40]


# # Animating a plot
# fig, ax = plt.subplots()
# plt.plot(numbersX, numbersY)
# line, = ax.plot(numbersX, numbersY, 'ro-')
# plt.ylim(0, 50)
# plt.xlim(0, 6)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Animating a Plot Example')
# plt.grid()
# def animate(i):
#     line.set_ydata([y + i for y in numbersY])
#     return line,
# ani = animation.FuncAnimation(fig, animate, frames=10, interval=1, blit=True)
# plt.show()

filepath = os.path.join(os.path.dirname(__file__), 'electric_vehicle_analytics.csv')
df = pd.read_csv(filepath)
print(df.head())

# Make sure to add spacing between each x label

fig, ax = plt.subplots(figsize=(14, 8))
plt.bar(df['Make'], df['Battery_Capacity_kWh'] , color='skyblue', width=0.4, align='center')
plt.xlabel('Make')
plt.ylabel('Battery Capacity (kWh)')
plt.title('Electric Vehicle Battery Capacity by Make')

plt.show()


fig, ax = plt.subplots(figsize=(14, 8))
plt.bar(df['Make'], df['Range_km'] , color='skyblue', width=0.4, align='center')
plt.xlabel('Make')
plt.ylabel('Range (km)')
plt.title('Electric Vehicle Range by Make')

plt.show()
max = df['Range_km'].max()
model = df[df['Range_km'] == max]['Model'].values[0]
# The line above works by filtering the DataFrame to find the row where 'Range_km' equals the maximum value, then selecting the 'Model' column from that row and extracting the value.
# print(f'Max Range: {max} km from {model}')
make = df[df['Range_km'] == max]['Make'].values[0]
# Filtering for the max and the Make column produces a Series, so we use .values[0] to get the actual string value.
print(f'Max Range: {max} km from {make} {model}')

fig, ax = plt.subplots(figsize=(14, 8))
plt.scatter(df['Battery_Capacity_kWh'], df['Charging_Time_hr'], color='skyblue')
plt.xlabel('Battery Capacity (kWh)')
plt.ylabel('Charging Time (hrs)')
plt.title('Battery Capacity vs Charging Time')

plt.show()

fig, ax = plt.subplots(figsize=(14,8))
# We want to filter the y-axis to a mean for each make
mean_charging_time = df.groupby('Make')['Charging_Time_hr'].mean().reset_index()
'''
Step by step, this code groups the DataFrame by 'Make', calculates the mean of 'Charging_Time_hr' for each group, and then resets the index to turn the result back into a DataFrame.
The result is a new DataFrame with two columns: 'Make' and the mean 'Charging_Time_hr'.
This DataFrame can then be used for plotting.
reset_index() is used to convert the Series back into a DataFrame from the groupby operation.
Groupby creates a Series with 'Make' as the index and the mean 'Charging_Time_hr' as values. reset_index() makes 'Make' a column again.
'''
plt.bar(mean_charging_time['Make'], mean_charging_time['Charging_Time_hr'], color='green', width = 0.5)
plt.ylim(1.05, 1.35)
plt.xlabel("Make")
plt.ylabel("Electriity Cost per KWh")
plt.title("Make vs Cost of Electricity")
plt.show()