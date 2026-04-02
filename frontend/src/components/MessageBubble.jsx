import AnswerPlayer from './AnswerPlayer'

export default function MessageBubble({ message, onPause, onFinished, onResumeFinished }) {
  const isUser = message.role === 'user'

  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        {isUser && <span>{message.content}</span>}

        {!isUser && message.status === 'thinking' && (
          <span className="thinking">Thinking...</span>
        )}

        {!isUser && message.status === 'playing' && (
          <AnswerPlayer
            streamingWords={message.streamingWords}
            isStreaming={message.isStreaming}
            sentences={message.sentences}
            words={message.words}
            wordToSentenceMap={message.wordToSentenceMap}
            onPause={(pauseData) => onPause?.({ ...pauseData, messageId: message.id, ...message })}
            onFinished={() => onFinished?.(message.id)}
          />
        )}

        {!isUser && message.status === 'completed' && (
          <span>{message.text || message.content}</span>
        )}

        {!isUser && message.status === 'paused' && (
          <div className="answer-player">
            <div className="text-container">
              {message.words?.map((w, i) => {
                let style = {}
                if (i < (message.pauseData?.wordIndex || 0)) {
                  style = { opacity: 1, color: '#ffffff' }
                } else if (i < (message.pauseData?.revealIndex || 0)) {
                  style = { opacity: 0.4, color: '#888888' }
                } else {
                  style = { opacity: 0 }
                }
                return <span key={i} style={style}>{w} </span>
              })}
            </div>
            <div style={{ textAlign: 'center', marginTop: 12 }}>
              <span className="paused-badge">⏸ Paused — will resume after next answer</span>
            </div>
          </div>
        )}

        {!isUser && message.status === 'resuming' && (
          <>
            <p className="thinking" style={{ marginBottom: 8 }}>Resuming previous answer...</p>
            <AnswerPlayer
              words={message.words}
              sentences={message.sentences}
              wordToSentenceMap={message.wordToSentenceMap}
              startSentence={message.pauseData?.sentenceIndex || 0}
              startWord={message.pauseData?.wordIndex || 0}
              isResume={true}
              onPause={(pauseData) => onPause?.({ ...pauseData, messageId: message.id, ...message })}
              onFinished={() => onResumeFinished?.(message.id)}
            />
          </>
        )}
      </div>
    </div>
  )
}
