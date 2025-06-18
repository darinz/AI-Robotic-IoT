# IMPLEMENTATION DETAILS: RL - USING REINFORCEMENT LEARNING TO BUILD THE SELF-DRIVING SYSTEM

Our implementation integrates a range of hardware components, software packages, and machine learning techniques to create a reinforcement learning-based self-driving robotic vehicle. 

Below are the core elements of our system:

## Hardware Components

At the core of the system is the Raspberry Pi 4, which serves as the central processing unit. This compact yet powerful single-board computer is responsible for running all aspects of the control logic, including the reinforcement learning algorithms. While its quad-core processor and GPIO compatibility make it suitable for basic robotics applications, its lack of GPU acceleration and limited RAM posed significant challenges for more computationally intensive tasks such as deep learning.

The car relies on an ultrasonic sensor for real-time obstacle detection. Mounted at the front of the chassis, the sensor continuously measures the distance between the car and nearby objects. These measurements are fed into the RL agent's state representation, enabling the car to make informed decisions about whether to move forward or take evasive action. The simplicity and reliability of ultrasonic sensing made it a cornerstone of the navigation strategy, especially in the absence of a working vision system.

A Raspberry Pi Camera Module was also integrated into the hardware design to enable vision-based navigation. In theory, this camera would allow the car to perceive its environment in richer detail, offering a path toward more complex reinforcement learning with visual input. However, in practice, the camera frequently failed to initialize properly or overwhelmed the limited processing capabilities of the Raspberry Pi, rendering it unreliable during live training sessions. As a result, it was considered optional and was not actively used in the deployed system.

Motion is handled by a motor driver module, which interfaces with the Raspberry Pi’s GPIO pins to control two rear-wheel DC motors. This motor driver translates logic-level signals into appropriate voltage and current levels to drive the motors forward, backward, or steer the car left and right. The system supports four basic movements—forward, backward, left turn, and right turn—which collectively enable the agent to navigate a variety of obstacle configurations.

The motors and car chassis form the physical platform upon which the entire system operates. The chassis is lightweight and compact, providing the maneuverability needed for indoor training environments. The placement of the drive motors at the rear allows for differential steering, which is sufficient for basic navigation tasks.

Finally, the system is powered by a battery pack, which supplies power to both the Raspberry Pi and the motors. Managing power distribution effectively was crucial to ensure stable operation during extended training sessions. Care was taken to balance performance and energy efficiency, especially since unexpected shutdowns or low-voltage conditions could lead to corrupted models or unstable behavior during learning.

Together, these components form a tightly integrated embedded system designed for real-time interaction with a physical environment. While minimal in terms of cost and complexity, the hardware stack is sufficient to support foundational self-driving functionality and explore core reinforcement learning principles in a real-world setting.

## Software and Libraries

The software stack for this project was built primarily in Python 3.x, chosen for its simplicity, rich ecosystem of libraries, and compatibility with the Raspberry Pi. Python provided an accessible and flexible foundation for developing both the control logic and the reinforcement learning framework, allowing us to iterate quickly and focus on the core logic of the self-driving agent.

To interface with the hardware components, we used the RPi.GPIO library. This low-level library enabled direct control over the Raspberry Pi’s GPIO pins, which were essential for sending signals to the motor driver and reading data from the ultrasonic sensor. With RPi.GPIO, we were able to translate the agent’s decisions into physical movement commands and gather real-time feedback from the environment, creating a closed-loop control system.

Although the Raspberry Pi Camera Module presented reliability issues, we experimented with OpenCV (cv2) for potential vision-based input. OpenCV was used for image capture and basic preprocessing, such as grayscale conversion and resizing. While we were unable to fully integrate visual perception into the RL pipeline due to hardware limitations, these early explorations laid the groundwork for future work in vision-based navigation.

For the implementation of reinforcement learning, particularly the Q-learning algorithm, NumPy played a central role. It was used to create and manipulate the Q-table, which represents the agent's learned policy as a matrix of state-action values. NumPy’s efficient array operations allowed us to perform updates and lookups in real time, enabling the agent to learn and adapt on-the-fly as it interacted with its environment.

In the early stages, we also experimented with PyTorch, a popular deep learning library, to implement a Deep Q-Network (DQN). PyTorch allowed us to define neural network architectures, perform gradient-based optimization, and manage model weights. However, due to the Raspberry Pi’s limited processing power and lack of GPU support, we were unable to run DQN training effectively. The DQN implementation was therefore shelved in favor of the more computationally lightweight tabular Q-learning method.

To ensure that learning progress was not lost between sessions, we utilized Python’s pickle module for model persistence. Q-tables (and DQN weights, when applicable) were serialized and saved to disk at regular intervals. Upon restart, the system would load these saved values, allowing the agent to resume training from its previous state. This long-term memory was critical for enabling cumulative learning across multiple days of training and testing.

Overall, the software architecture combines low-level hardware control with higher-level reinforcement learning logic. While constrained by the Raspberry Pi’s hardware limitations, the software stack proved effective for implementing and evaluating RL-based navigation in a compact, real-world setting.

## Reinforcement Learning Algorithms

To enable autonomous decision-making in a dynamic environment, we implemented two reinforcement learning (RL) approaches: Q-Learning and Deep Q-Networks (DQN). Each algorithm was selected based on the nature of the state space and the computational limitations of the hardware.

Q-Learning was the primary algorithm used in our implementation, especially in the stages of development and deployment. It is a model-free, value-based RL method that relies on a discrete and finite set of states and actions. In our system, the state space was simplified using quantized distance bins derived from the ultrasonic sensor readings—for example, discretizing the sensed distance into categories like "very close," "moderate," or "far." These bins formed the basis of a Q-table, where each entry represents the expected utility (Q-value) of taking a specific action from a given state. The Q-values were iteratively updated as the car explored its environment, using the standard Q-learning update rule. This approach proved feasible and reliable given the limited processing capabilities of the Raspberry Pi, especially when paired with efficient NumPy operations.

As we considered scaling the system to handle more complex input modalities, such as raw images from the camera or continuous distance values, we explored Deep Q-Networks (DQN). DQN replaces the tabular Q-function with a deep neural network that estimates Q-values for each possible action given a high-dimensional input. For instance, instead of looking up a Q-value in a table, the network could take in a preprocessed camera frame or a continuous distance vector and predict the expected reward for each movement direction. We used PyTorch to define and train the DQN model, leveraging feedforward neural network architectures.

However, due to the hardware constraints of the Raspberry Pi, particularly its limited memory and lack of GPU support, running DQN training in real time was not feasible. The computational load of forward passes, backpropagation, and replay buffer management was too high for the system to handle efficiently. As a result, while the DQN approach was theoretically implemented and partially tested, we ultimately relied on the more lightweight Q-learning algorithm for live experimentation.

Together, these algorithms reflect the trade-offs between expressiveness and practicality. Q-learning was simple and efficient enough to operate within our hardware constraints, while DQN offered a path toward richer and more scalable solutions—albeit one that would require more capable hardware to realize fully.

## Advanced Reinforcement Learning Techniques

To improve learning efficiency and agent performance, we incorporated several advanced reinforcement learning techniques into our system. These enhancements helped address common RL challenges such as sparse rewards, unstable training, and exploration-exploitation trade-offs.

Reward Shaping was a key technique we used to accelerate the learning process and encourage safer driving behavior. In traditional RL setups, agents often receive rewards only for terminal outcomes, such as reaching a goal or crashing. However, this sparse feedback can make learning slow and inefficient. To address this, we crafted a more nuanced reward function that provided continuous feedback throughout the car’s operation. Specifically, the agent received a reward of +1 for moving forward without a collision, which reinforced proactive behavior. Actions such as reversing or turning without making progress were penalized with a -1, discouraging unnecessary or unproductive movements. A significant penalty of -10 was applied for any collision, strongly discouraging reckless driving. Additionally, a small positive reward of +0.5 was given when the car approached open paths or successfully avoided obstacles, which helped guide the agent toward strategic behaviors like making gradual turns and preferring safe, obstacle-free routes. This layered reward structure helped shape the agent’s learning trajectory toward more effective and cautious navigation strategies.

In our Deep Q-Network (DQN) implementation, we also integrated Experience Replay, a technique essential for stabilizing training in neural network-based RL models. Without experience replay, updates to the Q-network are highly correlated with the most recent experiences, leading to instability and inefficient learning. To mitigate this, we maintained a replay buffer that stored past experiences in the form of tuples: (state, action, reward, next_state, done). During training, the network randomly sampled mini-batches from this buffer rather than learning only from the latest step. This broke the temporal correlations between consecutive observations and allowed the model to learn from a more diverse distribution of experiences. While this method significantly improves the learning process, it also requires more memory and processing power, which ultimately limited its usage on the Raspberry Pi due to hardware constraints.

Finally, we employed an ε-greedy exploration strategy to balance the trade-off between exploring new actions and exploiting known good ones. At the beginning of training, the agent selected actions at random with a high probability (ε = 1.0), allowing it to explore a wide range of behaviors. Over time, this exploration probability gradually decayed toward ε = 0.1, shifting the focus toward exploitation of learned policies. This decaying ε strategy ensured that the agent had ample opportunity to discover effective strategies early on while converging to more consistent and optimal behavior as training progressed.

Together, these advanced techniques—reward shaping, experience replay, and exploration scheduling—provided a solid foundation for effective and stable reinforcement learning, even within the constrained computing environment of the Raspberry Pi. They enabled the car to not only learn from direct outcomes but also develop a deeper, more refined understanding of how different actions affected long-term success.

## Persistence and Online Learning

One of the key design considerations in our reinforcement learning system was the ability to preserve the agent’s learning progress across multiple training sessions. To accomplish this, we implemented persistent storage for the RL model—whether it was a Q-table for tabular Q-learning or neural network weights for the Deep Q-Network (DQN) variant. During operation, the model is periodically saved to disk using serialization techniques (e.g., via Python’s pickle module). This checkpointing mechanism ensures that the valuable knowledge accumulated by the agent over time is not lost due to power failures, crashes, or simple reboots. When the program is launched, it automatically checks for the presence of a saved model and loads it into memory if available, allowing the car to resume learning from its previous state rather than starting from scratch.

In addition to persistence, the system is designed to support online learning, meaning the agent continues to refine its policy in real time as it interacts with the environment. Every experience—whether it's a successful avoidance of an obstacle or a collision—immediately influences the learning process. This continuous feedback loop allows the agent to adapt dynamically to changes in the environment or unexpected scenarios, making it more robust over time. Unlike offline training, where learning occurs in isolated batches based on pre-collected data, online learning enables the system to respond to real-world feedback as it happens, which is particularly important in physical systems like autonomous vehicles operating in unpredictable settings.

Together, persistence and online learning form a powerful combination. Persistence enables long-term knowledge accumulation across days or even weeks of training, while online learning ensures that the agent continually evolves and adapts its decision-making strategy in real-time. These capabilities are crucial for building an intelligent, self-improving system that can learn to navigate complex environments despite the limited computational resources of the Raspberry Pi platform.

## Data Structures and Protocols

To enable the reinforcement learning agent to interact intelligently with its environment, we developed a structured approach for managing state information, defining possible actions, assigning rewards, and storing past experiences. These components form the backbone of the learning framework and directly impact the agent's ability to learn effectively over time.

State Representation varied depending on the learning algorithm in use. For traditional Q-learning, which relies on a tabular representation, we discretized the environment by quantizing the ultrasonic sensor readings into bins (e.g., 'very close', 'close', 'far'). This simplification made the state space manageable and well-suited to the limited memory and processing power of the Raspberry Pi. In contrast, for Deep Q-Network (DQN)-based learning, we used a more complex state representation that could include continuous values such as raw distance measurements and even camera frame data (though the camera was often unreliable in practice). These high-dimensional inputs were passed as vectors to the neural network for Q-value approximation.

The action space was defined as a discrete set of basic motor commands the car could perform. These included 'forward', 'backward', 'left', and 'right'. This abstraction made it easier to interface motor control with the RL decision-making logic. Each action triggered a specific combination of GPIO signals through the motor driver, which in turn controlled the direction and behavior of the car.

The reward function played a critical role in shaping the agent’s behavior. Rather than relying solely on simple binary feedback (success vs. failure), we fine-tuned the reward structure to encourage smoother and more intelligent navigation. For example, moving forward without encountering an obstacle yielded a small positive reward, while collisions triggered a substantial penalty. Minor penalties were also applied for reversing or turning aimlessly, discouraging inefficient or overly cautious behavior. Subtle positive rewards were given when the agent moved toward clearer paths, helping the agent learn more strategic, proactive navigation patterns.

For the DQN implementation, we included a replay buffer to store the agent’s experiences in the form of (state, action, reward, next_state, done) tuples. This buffer was implemented using an efficient circular queue or deque structure, which enabled fast insertion and retrieval of recent experiences. During training, the model sampled random mini-batches from this buffer, allowing it to learn from a diverse and de-correlated set of experiences. This mechanism significantly stabilized learning and made the model less sensitive to transient environmental patterns.

Together, these data structures and protocols created a coherent framework for real-time decision-making, enabling our system to operate with both efficiency and adaptability despite the hardware constraints of the platform.

### References and Citations

We referenced the following foundational works and technical documentation:
1.	Watkins & Dayan (1992) – Q-learning
2.	Mnih et al., (2015) – Human-level control through deep reinforcement learning (DQN), Nature
3.	Sutton & Barto – Reinforcement Learning: An Introduction (2nd Edition)
4.	Raspberry Pi Documentation – https://www.raspberrypi.com/documentation/
5.	PyTorch Docs – https://pytorch.org/docs/
6.	OpenCV Docs – https://docs.opencv.org/