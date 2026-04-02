import { useState, useRef } from 'react'

export default function Sidebar({ onPdfUploaded }) {
  const [status, setStatus] = useState(null) // null | 'loading' | 'success' | 'error'
  const [statusText, setStatusText] = useState('')
  const fileRef = useRef()

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setStatus('loading')
    setStatusText('Processing PDF...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData })
      const data = await res.json()

      if (data.error) {
        setStatus('error')
        setStatusText(data.error)
      } else {
        setStatus('success')
        setStatusText(`Indexed ${data.chunks} chunks`)
        onPdfUploaded()
      }
    } catch (err) {
      setStatus('error')
      setStatusText('Upload failed: ' + err.message)
    }
  }

  return (
    <div className="sidebar">
      <h2>RAG Chatbot</h2>
      <div>
        <label>Upload PDF</label>
        <div className="upload-area" onClick={() => fileRef.current?.click()}>
          <input ref={fileRef} type="file" accept=".pdf" onChange={handleUpload} />
          <p style={{ color: '#8b949e', fontSize: 13 }}>Click to select a PDF</p>
        </div>
        {status && (
          <div className={`upload-status ${status}`}>{statusText}</div>
        )}
      </div>
    </div>
  )
}
