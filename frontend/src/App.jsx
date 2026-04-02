import { useState, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'

export default function App() {
  const [pdfReady, setPdfReady] = useState(false)
  const [messages, setMessages] = useState([])
  // LIFO stack of paused message IDs
  const [pauseStack, setPauseStack] = useState([])

  const handlePdfUploaded = useCallback(() => {
    setPdfReady(true)
    setMessages([])
    setPauseStack([])
  }, [])

  const handlePause = useCallback((state) => {
    // Push the paused message onto the stack
    setMessages(prev => prev.map(m =>
      m.status === 'playing' || m.status === 'resuming'
        ? { ...m, status: 'paused', pauseData: state }
        : m
    ))
    setPauseStack(prev => [...prev, state.messageId])
  }, [])

  const handleFinished = useCallback((messageId) => {
    // Mark this answer as completed
    setMessages(prev => prev.map(m =>
      m.id === messageId ? { ...m, status: 'completed' } : m
    ))

    // Pop the top of the pause stack and resume it
    setPauseStack(prev => {
      if (prev.length === 0) return prev
      const stack = [...prev]
      const topId = stack.pop()
      // Resume the most recently paused message
      setMessages(msgs => msgs.map(m =>
        m.id === topId && m.status === 'paused'
          ? { ...m, status: 'resuming' }
          : m
      ))
      return stack
    })
  }, [])

  // handleResumeFinished is the same as handleFinished —
  // when a resumed answer finishes, it may trigger the next resume from the stack
  const handleResumeFinished = handleFinished

  return (
    <div className="app">
      <Sidebar onPdfUploaded={handlePdfUploaded} />
      <ChatWindow
        pdfReady={pdfReady}
        messages={messages}
        setMessages={setMessages}
        onPause={handlePause}
        onFinished={handleFinished}
        onResumeFinished={handleResumeFinished}
      />
    </div>
  )
}
