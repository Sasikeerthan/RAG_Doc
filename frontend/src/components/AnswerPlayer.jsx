import { useRef, useEffect, useCallback, useState } from 'react'
import useTTS from '../hooks/useTTS'

/**
 * AnswerPlayer: streams text in gray, speaks sentences as they complete,
 * highlights words white via onboundary.
 *
 * Two modes:
 *   STREAMING: tokens arrive via SSE → words appear gray → completed sentences speak
 *   RESUME:    full data provided → renders from saved position
 */
export default function AnswerPlayer({
  streamingWords,
  isStreaming,
  sentences,
  words,
  wordToSentenceMap,
  startSentence = 0,
  startWord = 0,
  isResume = false,
  onPause,
  onFinished,
}) {
  const containerRef = useRef(null)
  const spanRefsRef = useRef([])
  const spokenUpToWordRef = useRef(isResume ? startWord : 0)
  const lastSpokenSentRef = useRef(isResume ? startSentence - 1 : -1)
  const [isPlaying, setIsPlaying] = useState(true)
  const finishedRef = useRef(false)

  // Streaming tracking
  const prevStreamCountRef = useRef(0)
  const queuedStreamSentsRef = useRef(0)
  const postStreamHandledRef = useRef(false)

  // ---- Highlight helpers ----
  const highlightUpTo = useCallback((globalWordIdx) => {
    const spans = spanRefsRef.current
    for (let i = spokenUpToWordRef.current; i <= globalWordIdx && i < spans.length; i++) {
      if (spans[i]) {
        spans[i].style.opacity = '1'
        spans[i].style.color = '#ffffff'
      }
    }
    spokenUpToWordRef.current = Math.max(spokenUpToWordRef.current, globalWordIdx + 1)
  }, [])

  // ---- TTS callbacks ----
  const onWordBoundary = useCallback((globalWordIdx) => {
    highlightUpTo(globalWordIdx)
  }, [highlightUpTo])

  const onSentenceEnd = useCallback((sentIdx) => {
    lastSpokenSentRef.current = sentIdx
  }, [])

  const onAllDone = useCallback(() => {
    if (finishedRef.current) return
    finishedRef.current = true
    // Highlight everything
    const spans = spanRefsRef.current
    for (let i = 0; i < spans.length; i++) {
      if (spans[i]) {
        spans[i].style.opacity = '1'
        spans[i].style.color = '#ffffff'
      }
    }
    setIsPlaying(false)
    onFinished?.()
  }, [onFinished])

  const tts = useTTS({ onWordBoundary, onSentenceEnd, onAllDone })

  // ================================================================
  // STREAMING: append word spans as they arrive
  // ================================================================
  useEffect(() => {
    if (isResume || !streamingWords) return
    const container = containerRef.current
    if (!container) return

    const prevCount = prevStreamCountRef.current
    for (let i = prevCount; i < streamingWords.length; i++) {
      const span = document.createElement('span')
      span.textContent = streamingWords[i] + ' '
      span.style.opacity = '0.4'
      span.style.color = '#888888'
      span.style.transition = 'opacity 0.15s, color 0.15s'
      container.appendChild(span)
      spanRefsRef.current.push(span)
    }
    prevStreamCountRef.current = streamingWords.length

    if (streamingWords.length % 5 === 0) {
      container.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [streamingWords, isResume])

  // ================================================================
  // STREAMING: detect completed sentences and enqueue to TTS
  // ================================================================
  useEffect(() => {
    if (isResume || !streamingWords || streamingWords.length === 0) return
    // Skip if post-stream is already handled
    if (postStreamHandledRef.current) return

    const text = streamingWords.join(' ')
    const sentenceRegex = /[^.!?]*[.!?]+/g
    const matches = []
    let m
    while ((m = sentenceRegex.exec(text)) !== null) {
      matches.push(m[0].trim())
    }

    const alreadyQueued = queuedStreamSentsRef.current
    for (let i = alreadyQueued; i < matches.length; i++) {
      const sentText = matches[i]
      if (sentText.length < 3) continue

      // Compute word range
      let wordStart = 0
      for (let s = 0; s < i; s++) {
        wordStart += matches[s].split(/\s+/).filter(w => w).length
      }
      const sentWordCount = sentText.split(/\s+/).filter(w => w).length
      const wordEnd = wordStart + sentWordCount - 1

      // Only speak if there's text after (confirming sentence is complete)
      const matchEnd = text.indexOf(sentText) + sentText.length
      const textAfter = text.substring(matchEnd).trim()
      if (textAfter.length > 0) {
        console.log('[AnswerPlayer] STREAM: enqueue sentence', i, 'words:', wordStart, '-', wordEnd)
        tts.enqueue(sentText, i, wordStart, wordEnd)
        queuedStreamSentsRef.current = i + 1
      }
    }
  }, [streamingWords, isResume, tts])

  // ================================================================
  // POST-STREAM: when done event arrives, enqueue remaining sentences
  // ================================================================
  useEffect(() => {
    if (isResume) return
    if (isStreaming) return
    if (!sentences || !sentences.length) return
    if (!wordToSentenceMap || !wordToSentenceMap.length) return
    if (postStreamHandledRef.current) return
    postStreamHandledRef.current = true

    // Build word ranges from the final map
    const ranges = {}
    wordToSentenceMap.forEach((sentIdx, wordIdx) => {
      if (!ranges[sentIdx]) ranges[sentIdx] = { start: wordIdx, end: wordIdx }
      else ranges[sentIdx].end = wordIdx
    })

    const alreadyQueued = queuedStreamSentsRef.current
    console.log('[AnswerPlayer] POST-STREAM: backend sentences:', sentences.length, 'already queued:', alreadyQueued)

    // Mark the last sentence so TTS knows when to fire onAllDone
    tts.markLast(sentences.length - 1)

    if (alreadyQueued >= sentences.length) {
      // All were spoken during streaming — wait for TTS to finish
      const checkDone = setInterval(() => {
        if (!window.speechSynthesis.speaking) {
          clearInterval(checkDone)
          onAllDone()
        }
      }, 200)
      return () => clearInterval(checkDone)
    }

    // Enqueue remaining sentences one by one (TTS hook chains them via onend)
    for (let i = alreadyQueued; i < sentences.length; i++) {
      const range = ranges[i]
      if (!range) continue
      console.log('[AnswerPlayer] POST-STREAM: enqueue sentence', i, 'words:', range.start, '-', range.end)
      tts.enqueue(sentences[i], i, range.start, range.end)
    }
  }, [isStreaming, sentences, wordToSentenceMap, isResume, tts, onAllDone])

  // ================================================================
  // RESUME MODE
  // ================================================================
  useEffect(() => {
    if (!isResume || !words || !words.length) return
    const container = containerRef.current
    if (!container) return

    const ranges = {}
    if (wordToSentenceMap) {
      wordToSentenceMap.forEach((sentIdx, wordIdx) => {
        if (!ranges[sentIdx]) ranges[sentIdx] = { start: wordIdx, end: wordIdx }
        else ranges[sentIdx].end = wordIdx
      })
    }

    for (let i = 0; i < words.length; i++) {
      const span = document.createElement('span')
      span.textContent = words[i] + ' '
      span.style.transition = 'opacity 0.15s, color 0.15s'
      if (i < startWord) {
        span.style.opacity = '1'
        span.style.color = '#ffffff'
      } else {
        span.style.opacity = '0.4'
        span.style.color = '#888888'
      }
      container.appendChild(span)
      spanRefsRef.current.push(span)
    }

    if (sentences) {
      tts.reset()
      tts.markLast(sentences.length - 1)
      for (let i = startSentence; i < sentences.length; i++) {
        const range = ranges[i]
        if (!range) continue
        if (i === startSentence && startWord > range.start) {
                const partialText = words.slice(startWord, range.end + 1).join(' ')
                tts.enqueue(partialText, i, startWord, range.end)
              } else {
                tts.enqueue(sentences[i], i, range.start, range.end)
              }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isResume])

  // ---- Raise Hand ----
  const handleRaiseHand = useCallback(() => {
    tts.cancel()
    setIsPlaying(false)

    onPause?.({
      sentenceIndex: lastSpokenSentRef.current + 1,
      wordIndex: spokenUpToWordRef.current,
      revealIndex: spanRefsRef.current.length,
    })
  }, [tts, onPause])

  return (
    <div className="answer-player">
      <div className="text-container" ref={containerRef} />
      {isPlaying && (
        <button className="raise-hand-btn" onClick={handleRaiseHand}>
          ✋ Raise Hand
        </button>
      )}
    </div>
  )
}
