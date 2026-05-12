import streamlit as st
import matplotlib.pyplot as plt

st.title("Training Visualizations")

epochs = [1,2,3,4,5]

loss = [1.2,0.9,0.6,0.4,0.2]

accuracy = [60,70,80,88,95]

fig, ax = plt.subplots()

ax.plot(epochs, loss)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curve")

st.pyplot(fig)

fig2, ax2 = plt.subplots()

ax2.plot(epochs, accuracy)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training Accuracy")

st.pyplot(fig2)