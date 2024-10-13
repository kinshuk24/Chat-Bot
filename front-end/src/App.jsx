import './App.css'
import LLMInterface from './Components/LLMInterface.tsx'
function App() {
  return (
    <div className="flex flex-col-reverse h-[10rem]">
      <div className="h-[50%]">
        <LLMInterface />
      </div>
    </div>
  )
}

export default App
