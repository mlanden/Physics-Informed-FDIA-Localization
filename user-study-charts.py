import matplotlib.pyplot as plt

# Define the data for the bars
x_values = ["Anit-\ndisassembly", "Obfuscation", "Anti-\ndebugger", "sandbox \nEvasion", "Combo", "depends on \nMalware"]
y_values = [5, 9, 3, 2, 2, 2]

# Define the colors for each bar
colors = ['#B85450', '#82B366', '#6C8EBF', '#D6B656', '#9673A6', '#666666']

# Create a bar chart with the x and y values and the specified colors
# plt.bar(x_values, y_values, color=colors)

# Add labels for the x and y axis and a title for the chart
# plt.xlabel('X Label')
# plt.ylabel("Number of Participants")

# Display the chart
# plt.savefig("challenges.pdf")
y_values = [2, 31, 6, 25, 0, 0]
plt.bar(x_values, y_values, color=colors[:4])
plt.ylabel("Number of Papers")
plt.savefig("papers.pdf")
# plt.show()