import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire

load_dotenv()

logfire.configure(send_to_logfire='if-token-present')

system_prompt="""You an experienced React developer. Create code that meets user's requirements."""

# Define the model
model = OpenAIModel(os.getenv('LLM_MODEL'), api_key=os.getenv('OPENAI_API_KEY'))

# Define the agent
agent = Agent(model=model, system_prompt=system_prompt)

user_prompt="""Create a functional React component that displays a user profile with the following details: name, email, and profile picture. 
Must use Zustand for state management and TailwindCSS for styling."""

# Run the agent
result = agent.run_sync(user_prompt=user_prompt)

print(result.data)

### Output:

# 8:16:22.720 agent run prompt=Create a functional React component that displays a user profi... use Zustand for state management and TailwindCSS for styling.
# 18:16:22.721   preparing model and tools run_step=1
# 18:16:22.721   model request
# 18:16:36.901   handle model response
# To create a functional React component that displays a user profile with the specified details using Zustand for state management and TailwindCSS for styling, follow the steps below:

# First, ensure that you have the required packages installed in your React project:

# ```bash
# npm install zustand tailwindcss
# ```

# Now, set up Zustand for state management. Create a store file for managing user data, let's call it `useUserStore.js`:

# ```javascript
# // useUserStore.js
# import create from 'zustand';

# const useUserStore = create((set) => ({
#   user: {
#     name: 'John Doe',
#     email: 'john.doe@example.com',
#     profilePicture: 'https://example.com/profile.jpg', // Replace with an actual image URL
#   },
#   setUser: (newUser) => set({ user: newUser }),
# }));

# export default useUserStore;
# ```

# Next, create the `UserProfile` component that utilizes this store:

# ```javascript
# // UserProfile.js
# import React from 'react';
# import useUserStore from './useUserStore';

# const UserProfile = () => {
#   const { user } = useUserStore();

#   return (
#     <div className="flex items-center bg-white rounded-lg shadow p-6 max-w-md mx-auto">
#       <img
#         src={user.profilePicture}
#         alt={`${user.name}'s profile`}
#         className="w-24 h-24 rounded-full mr-4"
#       />
#       <div className="flex flex-col">
#         <h2 className="text-xl font-semibold text-gray-800">{user.name}</h2>
#         <p className="text-gray-600">{user.email}</p>
#       </div>
#     </div>
#   );
# };

# export default UserProfile;
# ```

# Lastly, make sure you incorporate the `UserProfile` component into your main application file, typically `App.js`, like so:

# ```javascript
# // App.js
# import React from 'react';
# import UserProfile from './UserProfile';
# import './App.css'; // Make sure to import your TailwindCSS styles

# function App() {
#   return (
#     <div className="min-h-screen bg-gray-100 flex items-center justify-center">
#       <UserProfile />
#     </div>
#   );
# }

# export default App;
# ```

# ### TailwindCSS Setup

# Make sure TailwindCSS is properly set up in your project. If not, you can follow the [TailwindCSS installation guide](https://tailwindcss.com/docs/installation) to add it to your project.

# ### Summary

# With the above setup, you have a functional user profile component styled with TailwindCSS and state managed using Zustand. The component fetches the user data from the Zustand store and displays the user profile picture, name, and email. You can easily update the state using the `setUser` function from the Zustand store if needed.