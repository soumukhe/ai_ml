import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire

load_dotenv()

logfire.configure(send_to_logfire='if-token-present')

system_prompt="""You are an experienced React developer tasked with creating reusable, scalable, and high-performance components. Your primary goal is to ensure that each component adheres to the latest best practices in React development.

Please follow these guidelines when developing each component:

1. **Component Naming:** Use descriptive names for your components (e.g., `UserProfile`, `ProjectList`) that clearly indicate their purpose.
2. **Props Management:**
   - Define props with appropriate types using TypeScript or PropTypes to ensure type safety and clarity.
   - Pass only necessary props down from parent components to avoid unnecessary re-renders.
3. **State Management:** Use React's built-in state management (e.g., `useState`) for local component states, and consider context APIs or Redux for global state if needed.
4. **Lifecycle Methods:**
   - Utilize functional components with hooks (`useEffect`, `useContext`, etc.) instead of class-based components to take advantage of modern React features.
5. **Performance Optimization:**
   - Implement memoization techniques (e.g., `React.memo`) for pure components that don't need re-renders on prop changes.
   - Use virtual lists or pagination for large datasets to improve rendering performance.
6. **Styling:** Apply consistent styling using CSS-in-JS solutions like styled-components, emotion, or classnames. Avoid inline styles unless absolutely necessary.
7. **Accessibility:**
   - Ensure that components are accessible by following WCAG guidelines and using appropriate ARIA attributes where needed.
8. **Documentation:**
   - Create clear and concise documentation for each component, including usage examples, props descriptions, and any other relevant information.
   - Create a style guide or component library to maintain consistency across the application.
   - Create a README file for each component with installation instructions, usage guidelines, and other relevant details.
9. **Testing:** Write unit tests for your components using testing frameworks like Jest or React Testing Library to ensure they function correctly under various scenarios.
10. **Code Organization:**
    - Keep related files organized in a consistent directory structure (e.g., `components/`, `hooks/`).
    - Use meaningful folder names and subfolders if necessary to maintain clarity.
11.   **Error Handling:**
    - Implement error boundaries to catch and handle errors within your components gracefully.
12. **Logging:**
    - Use appropriate logging mechanisms (e.g., `console.log`, `sentry`) to track component behavior and debug issues effectively.
13.   **Performance Monitoring:**
    - Integrate performance monitoring tools (e.g., React Profiler, Lighthouse) to identify and address performance bottlenecks in your components.
14. **Security:**
    - Sanitize user inputs and avoid direct DOM manipulation to prevent security vulnerabilities like XSS attacks.
15. **Code Comments:**
    - Add comments to explain complex logic, algorithms, or workarounds in your components for better code readability.

**Additional Notes:**
- Ensure that all components are modular, allowing for easy reuse across different parts of the application.
- Maintain clean code with proper indentation, spacing, and comments where appropriate.
- Follow best practices in error handling and logging within your components.

Return the component code, styled with TailwindCSS, that meets the requirements outlined above. 
You can use Zustand for state management in your components. 
Return the Zustand store code. Return the README file for the component with installation instructions, usage guidelines, and other relevant details."""

# Define the model
model = OpenAIModel(os.getenv('LLM_MODEL'), api_key=os.getenv('OPENAI_API_KEY'))

# Define the agent
agent = Agent(model=model, system_prompt=system_prompt)

user_prompt="""Create a functional React component that displays a user profile with the following details: name, email, and profile picture. 
Must use Zustand for state management and TailwindCSS for styling."""

result = agent.run_sync(user_prompt=user_prompt)

print(result.data)

##### Output:
# 18:21:51.652 agent run prompt=Create a functional React component that displays a user profi... use Zustand for state management and TailwindCSS for styling.
# 18:21:51.653   preparing model and tools run_step=1
# 18:21:51.653   model request
# 18:22:11.155   handle model response
# Here’s how to create a functional React component to display a user profile with Zustand for state management and TailwindCSS for styling.

# ### 1. Zustand Store Code

# ```javascript
# // store/userStore.js
# import create from 'zustand';

# const useUserStore = create((set) => ({
#   user: {
#     name: '',
#     email: '',
#     profilePicture: '',
#   },
#   setUser: (userData) => set({ user: userData }),
# }));

# export default useUserStore;
# ```

# ### 2. UserProfile Component Code

# ```javascript
# // components/UserProfile.tsx
# import React from 'react';
# import useUserStore from '../store/userStore';

# // Define the props type
# interface UserProfileProps {
#   name: string;
#   email: string;
#   profilePicture: string;
# }

# // UserProfile Component
# const UserProfile: React.FC<UserProfileProps> = () => {
#   // Accessing the user state from Zustand store
#   const { user } = useUserStore();

#   return (
#     <div className="flex flex-col items-center p-4 bg-white rounded-lg shadow-md">
#       <img
#         src={user.profilePicture || 'https://via.placeholder.com/150'}
#         alt={`${user.name}'s profile`}
#         className="w-32 h-32 rounded-full border-2 border-gray-300"
#       />
#       <h2 className="mt-4 text-xl font-semibold">{user.name || 'User Name'}</h2>
#       <p className="mt-2 text-gray-600">{user.email || 'user@example.com'}</p>
#     </div>
#   );
# };

# export default UserProfile;
# ```

# ### 3. README File for UserProfile Component

# ```markdown
# # UserProfile Component

# The `UserProfile` component displays a user's profile information including name, email, and profile picture. It utilizes Zustand for state management and TailwindCSS for styling.

# ## Installation

# To use the `UserProfile` component in your React application, follow the steps below:

# 1. **Install Zustand**

#    If you haven’t already installed Zustand, you can do so by running:

#    ```bash
#    npm install zustand
#    ```

# 2. **Install TailwindCSS**

#    Ensure TailwindCSS is set up in your project. You can follow the [TailwindCSS installation guide](https://tailwindcss.com/docs/installation) for setup instructions.

# 3. **Create UserStore**

#    Create a Zustand store for user data. The store should be created in a file named `userStore.js` inside a `store` directory:

#    ```javascript
#    // store/userStore.js
#    import create from 'zustand';

#    const useUserStore = create((set) => ({
#      user: {
#        name: '',
#        email: '',
#        profilePicture: '',
#      },
#      setUser: (userData) => set({ user: userData }),
#    }));

#    export default useUserStore;
#    ```

# 4. **Add UserProfile Component**

#    Create a file named `UserProfile.tsx` inside the `components` directory and add the following code:

#    ```javascript
#    // components/UserProfile.tsx
#    import React from 'react';
#    import useUserStore from '../store/userStore';

#    interface UserProfileProps {}

#    const UserProfile: React.FC<UserProfileProps> = () => {
#      const { user } = useUserStore();

#      return (
#        <div className="flex flex-col items-center p-4 bg-white rounded-lg shadow-md">
#          <img
#            src={user.profilePicture || 'https://via.placeholder.com/150'}
#            alt={`${user.name}'s profile`}
#            className="w-32 h-32 rounded-full border-2 border-gray-300"
#          />
#          <h2 className="mt-4 text-xl font-semibold">{user.name || 'User Name'}</h2>
#          <p className="mt-2 text-gray-600">{user.email || 'user@example.com'}</p>
#        </div>
#      );
#    };

#    export default UserProfile;
#    ```

# ## Usage

# To use the `UserProfile` component, you need to set the user data in the Zustand store. Here's an example of how to do this:

# ```javascript
# import React from 'react';
# import useUserStore from './store/userStore';
# import UserProfile from './components/UserProfile';

# const App = () => {
#   const setUser = useUserStore((state) => state.setUser);

#   const setUserData = () => {
#     setUser({ 
#       name: 'John Doe', 
#       email: 'john.doe@example.com', 
#       profilePicture: 'https://example.com/john.jpg' 
#     });
#   };

#   return (
#     <div className="flex items-center justify-center min-h-screen bg-gray-100">
#       <button onClick={setUserData} className="px-4 py-2 mb-4 text-white bg-blue-500 rounded hover:bg-blue-600">
#         Load User Profile
#       </button>
#       <UserProfile />
#     </div>
#   );
# };

# export default App;
# ```

# ## Notes

# - Make sure to replace the default values in the component when using actual data.
# - Ensure that the Zustand store is set up correctly to manage user data effectively.
# ```

# This setup creates a reusable `UserProfile` component that fetches and displays user information, styled using TailwindCSS, and manages its state with Zustand.