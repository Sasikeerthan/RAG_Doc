import { useState, useRef, useEffect, useCallback } from 'react'
import MessageBubble from './MessageBubble'

let nextId = 1

export default function ChatWindow({
  pdfReady,
  messages,
  setMessages,
  onPause,
  onFinished,
  onResumeFinished,
}) {
  const [input, setInput] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const messagesEndRef = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault()
    const question = input.trim()
    if (!question || !pdfReady || isAsking) return

    setInput('')
    setIsAsking(true)

    const userId = nextId++
    const assistantId = nextId++

    // Add user message + thinking placeholder
    setMessages(prev => [
      ...prev,
      { id: userId, role: 'user', content: question },
      {
        id: assistantId,
        role: 'assistant',
        status: 'thinking',
        streamingWords: [],
        isStreaming: true,
        sentences: null,
        words: null,
        wordToSentenceMap: null,
        text: '',
      },
    ])

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let accumulatedText = ''
      let accumulatedWords = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Parse SSE events from buffer
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line

        let eventType = null
        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim()
          } else if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim()
            if (!dataStr) continue

            try {
              const data = JSON.parse(dataStr)

              if (eventType === 'token') {
                accumulatedText += data.token
                accumulatedWords = accumulatedText.split(/\s+/).filter(w => w.length > 0)

                setMessages(prev => prev.map(m =>
                  m.id === assistantId
                    ? { ...m, status: 'playing', streamingWords: [...accumulatedWords] }
                    : m
                ))
              } else if (eventType === 'done') {
                setMessages(prev => prev.map(m =>
                  m.id === assistantId
                    ? {
                        ...m,
                        isStreaming: false,
                        text: data.text,
                        sentences: data.sentences,
                        words: data.words,
                        wordToSentenceMap: data.wordToSentenceMap,
                      }
                    : m
                ))
              }
            } catch {
              // ignore malformed JSON
            }
            eventType = null
          }
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, status: 'completed', text: 'Error: ' + err.message }
          : m
      ))
    }

    setIsAsking(false)
  }, [input, pdfReady, isAsking, setMessages])

  return (
    <div className="chat-area">
      <div className="chat-header">
        <h1>RAG Chatbot</h1>
        <p>Streaming text + audio · Raise Hand to pause · auto-resume</p>
      </div>

      <div className="messages">
        {messages.length === 0 && pdfReady && (
          <p style={{ color: '#8b949e', textAlign: 'center', marginTop: 40 }}>
            PDF indexed. Ask a question to get started.
          </p>
        )}
        {!pdfReady && (
          <p style={{ color: '#8b949e', textAlign: 'center', marginTop: 40 }}>
            Upload a PDF to get started.
          </p>
        )}
        {messages.map(msg => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onPause={onPause}
            onFinished={onFinished}
            onResumeFinished={onResumeFinished}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            className="chat-input"
            type="text"
            placeholder={pdfReady ? 'Ask about your document...' : 'Upload a PDF first...'}
            value={input}
            onChange={e => setInput(e.target.value)}
            disabled={!pdfReady || isAsking}
          />
          <button
            className="chat-send-btn"
            type="submit"
            disabled={!pdfReady || isAsking || !input.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
