"use client";
import { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const CopyButton = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy!", err);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="flex items-center space-x-1 text-gray-400 hover:text-white transition-colors focus:outline-none"
    >
      {copied ? (
        <span className="text-emerald-400 flex items-center gap-1">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          <span className="text-[10px] uppercase font-bold">Copied!</span>
        </span>
      ) : (
        <span className="flex items-center gap-1">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
          </svg>
          <span className="text-[10px] uppercase font-bold">Copy</span>
        </span>
      )}
    </button>
  );
};

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Inicjalizacja sesji
  useEffect(() => {
    let id = sessionStorage.getItem('chat_session_id');
    if (!id) {
      id = uuidv4();
      sessionStorage.setItem('chat_session_id', id);
    }
    setSessionId(id);
  }, []);

  // Auto-scroll do najnowszej wiadomości
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Funkcja nowej sesji
  const handleNewChat = () => {
    const newId = uuidv4();
    sessionStorage.setItem('chat_session_id', newId);
    setSessionId(newId);
    setMessages([]);
    setInput('');
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { id: uuidv4(), role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    const assistantMsgId = uuidv4();
    setMessages(prev => [...prev, { id: assistantMsgId, role: 'assistant', content: '' }]);

    try {
      const response = await fetch(
        `/api/v1/ask?question=${encodeURIComponent(input)}&session_id=${sessionId}`
      );

      if (!response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';
      let lineBuffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        lineBuffer += decoder.decode(value, { stream: true });
        const lines = lineBuffer.split('\n');
        lineBuffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          try {
            const json = JSON.parse(line.substring(6));
            if (json.content) {
              accumulatedContent += json.content;
              setMessages(prev => prev.map(m =>
                m.id === assistantMsgId ? { ...m, content: accumulatedContent } : m
              ));
            }
          } catch (e) { /* Ignore incomplete chunks */ }
        }
      }
    } catch (error) {
      console.error("API connection error:", error);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#313338] text-[#dbdee1]">
      {/* Sidebar */}
      <div className="w-64 bg-[#2b2d31] flex flex-col hidden md:flex">
        <div className="p-4 shadow-md font-bold border-b border-[#1e1f22] flex justify-between items-center">
          <span>Conversations</span>
          <button
            onClick={handleNewChat}
            className="p-1 hover:bg-[#35373c] rounded text-gray-400 hover:text-white transition-all"
            title="New Chat"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
        </div>
        <div className="p-2 space-y-1 overflow-y-auto">
          <div className="bg-[#35373c] p-2 rounded text-white cursor-pointer flex items-center space-x-2">
            <span className="text-gray-400">#</span>
            <span>Main session</span>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-12 border-b border-[#1e1f22] flex items-center px-4">
          <span className="font-bold text-white flex items-center">
            <span className="text-gray-400 mr-2">#</span> UniLM Agent
          </span>
        </div>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map((msg) => (
            <div key={msg.id} className="flex items-start space-x-4 p-1 rounded group">
              <div className={`w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center font-bold text-white ${msg.role === 'user' ? 'bg-indigo-500' : 'bg-emerald-500'}`}>
                {msg.role === 'user' ? 'U' : 'A'}
              </div>
              <div className="flex-1 overflow-hidden">
                <div className="flex items-center space-x-2">
                  <span className={`font-medium ${msg.role === 'user' ? 'text-indigo-400' : 'text-emerald-400'}`}>
                    {msg.role === 'user' ? 'Student' : 'Agent'}
                  </span>
                </div>

                <div className="text-[#dbdee1] mt-1 leading-relaxed prose prose-invert max-w-none">
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
                      code({ node, inline, className, children, ...props }: any) {
                        const match = /language-(\w+)/.exec(className || '');
                        const codeContent = String(children).replace(/\n$/, '');

                        return !inline && match ? (
                          <div className="my-4 rounded-md overflow-hidden border border-[#1e1f22]">
                            {/* Toolbar z nazwą języka i przyciskiem kopiowania */}
                            <div className="bg-[#1e1f22] px-4 py-1.5 text-xs text-gray-400 font-mono border-b border-white/5 flex justify-between items-center">
                              <span>{match[1].toUpperCase()}</span>
                              <CopyButton text={codeContent} />
                            </div>
                            <SyntaxHighlighter
                              style={oneDark}
                              language={match[1]}
                              PreTag="div"
                              customStyle={{ margin: 0, padding: '1rem' }}
                              {...props}
                            >
                              {codeContent}
                            </SyntaxHighlighter>
                          </div>
                        ) : (
                          <code className="bg-[#2b2d31] px-1.5 py-0.5 rounded text-sm font-mono text-orange-300" {...props}>
                            {children}
                          </code>
                        );
                      }
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>

                  {isTyping && msg.role === 'assistant' && msg.content === '' && (
                    <div className="flex items-center text-gray-500 italic py-2">
                      <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                      <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.15s] mx-1"></span>
                      <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce"></span>
                      <span className="ml-2 text-xs">Thinking...</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={scrollRef} />
        </div>

        {/* Input Bar */}
        <div className="p-4 bg-[#313338]">
          <div className="bg-[#383a40] rounded-lg px-4 py-1 flex items-center shadow-md border-2 border-transparent focus-within:border-[#5865f2] transition-all">
            <button className="text-[#B5BAC1] hover:text-white pr-3 text-2xl font-light">+</button>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Ask agent."
              className="bg-transparent flex-1 outline-none py-3 text-[#dbdee1] placeholder:text-[#6D6F78]"
            />
            <button
              onClick={sendMessage}
              className="ml-2 p-2 text-gray-400 hover:text-indigo-400 transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 rotate-90" fill="currentColor" viewBox="0 0 24 24">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
          <div className="text-[10px] p-1 text-[#949BA4] mt-1 uppercase font-bold tracking-wider">
            Press <span className="hover:underline cursor-help">Enter</span> to send
          </div>
        </div>
      </div>
    </div>
  );
}