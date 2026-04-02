import { useRef, useEffect, useMemo } from 'react'

/**
 * TTS hook using the browser Web Speech API (SpeechSynthesis).
 * Speaks one sentence at a time to avoid Chrome's 15-second pause bug.
 * Uses onboundary for word highlighting, with timing-based fallback.
 */
export default function useTTS({ onWordBoundary, onSentenceEnd, onAllDone }) {
  const cbRef = useRef({ onWordBoundary, onSentenceEnd, onAllDone })
  useEffect(() => {
    cbRef.current = { onWordBoundary, onSentenceEnd, onAllDone }
  })

  const engine = useMemo(() => {
    const e = {
      // Playback queue: [{sentIdx, wordStart, wordEnd, text}]
      playQueue: [],
      speaking: false,
      cancelled: false,
      lastSentIdx: -1,
      currentUtterance: null,
      highlightTimer: null,

      /** Strip markdown so TTS reads clean text. */
      _cleanForTTS(text) {
        return text
          .replace(/\*\*([^*]+)\*\*/g, '$1')
          .replace(/\*([^*]+)\*/g, '$1')
          .replace(/#{1,6}\s*/g, '')
          .replace(/`([^`]+)`/g, '$1')
          .replace(/^\s*[-*]\s+/gm, '')
          .replace(/^\s*\d+\.\s+/gm, '')
          .replace(/\n+/g, ' ')
          .trim()
      },

      playNext() {
        if (e.cancelled) return
        if (e.playQueue.length === 0) {
          e.speaking = false
          return
        }

        const item = e.playQueue.shift()
        e.speaking = true
        const { sentIdx, wordStart, wordEnd, text } = item
        const isLast = sentIdx === e.lastSentIdx
        const wordCount = wordEnd - wordStart + 1

        const cleanText = e._cleanForTTS(text)
        if (!cleanText) {
          cbRef.current.onWordBoundary?.(wordEnd)
          cbRef.current.onSentenceEnd?.(sentIdx)
          if (isLast) {
            e.speaking = false
            cbRef.current.onAllDone?.()
          } else {
            e.playNext()
          }
          return
        }

        console.log('[TTS] speaking sentence', sentIdx, 'words:', wordStart, '-', wordEnd, 'isLast:', isLast)

        const utterance = new SpeechSynthesisUtterance(cleanText)
        e.currentUtterance = utterance

        // Try to use onboundary for word-level highlights
        let boundaryWorked = false
        let currentWord = wordStart

        utterance.onboundary = (ev) => {
          if (e.cancelled) return
          if (ev.name === 'word') {
            boundaryWorked = true
            if (currentWord <= wordEnd) {
              cbRef.current.onWordBoundary?.(currentWord)
              currentWord++
            }
          }
        }

        // Timing-based fallback: start after a short delay to see if onboundary fires
        const startTime = Date.now()
        let fallbackStarted = false

        const startFallback = (estimatedDuration) => {
          if (boundaryWorked || fallbackStarted || e.cancelled) return
          fallbackStarted = true
          const msPerWord = estimatedDuration / wordCount
          let fallbackWord = wordStart

          e.highlightTimer = setInterval(() => {
            if (e.cancelled) {
              clearInterval(e.highlightTimer)
              return
            }
            const elapsed = Date.now() - startTime
            const targetWord = wordStart + Math.floor(elapsed / msPerWord)
            while (fallbackWord <= Math.min(targetWord, wordEnd)) {
              cbRef.current.onWordBoundary?.(fallbackWord)
              fallbackWord++
            }
          }, 50)
        }

        // Check after 500ms if onboundary is firing; if not, start fallback
        setTimeout(() => {
          if (!boundaryWorked && !e.cancelled && e.speaking) {
            // Estimate ~150ms per word as rough speech rate
            const estimatedMs = wordCount * 150
            startFallback(estimatedMs)
          }
        }, 500)

        const finish = () => {
          clearInterval(e.highlightTimer)
          e.currentUtterance = null
          // Ensure all words in this sentence are highlighted
          cbRef.current.onWordBoundary?.(wordEnd)
          cbRef.current.onSentenceEnd?.(sentIdx)
          if (isLast) {
            e.speaking = false
            cbRef.current.onAllDone?.()
          } else {
            e.playNext()
          }
        }

        utterance.onend = () => {
          if (e.cancelled) return
          console.log('[TTS] ended sentence', sentIdx)
          finish()
        }

        utterance.onerror = (ev) => {
          if (e.cancelled) return
          console.log('[TTS] error sentence', sentIdx, ev.error)
          finish()
        }

        window.speechSynthesis.speak(utterance)
      },

      enqueue(text, sentIdx, wordStart, wordEnd) {
        if (e.cancelled) return
        e.playQueue.push({ sentIdx, wordStart, wordEnd, text })
        if (!e.speaking) {
          e.playNext()
        }
      },

      markLast(sentIdx) {
        e.lastSentIdx = sentIdx
      },

      cancel() {
        e.cancelled = true
        clearInterval(e.highlightTimer)
        e.playQueue = []
        e.speaking = false
        e.currentUtterance = null
        window.speechSynthesis.cancel()
      },

      reset() {
        e.cancelled = false
        e.playQueue = []
        e.speaking = false
        e.lastSentIdx = -1
      },
    }
    return e
  }, [])

  useEffect(() => {
    return () => engine.cancel()
  }, [engine])

  return engine
}
