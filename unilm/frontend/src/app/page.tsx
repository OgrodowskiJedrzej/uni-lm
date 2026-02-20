"use client";
import { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

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

  useEffect(() => {
    let id = localStorage.getItem('chat_session_id');
    if (!id) {
      id = uuidv4();
      localStorage.setItem('chat_session_id', id);
      console.log(id);
    }
    setSessionId(id);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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
        `http://localhost:8000/api/v1/ask?question=${encodeURIComponent(input)}&session_id=${sessionId}`
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
          } catch (e) {
            // Ignore incomplete JSON chunks
          }
        }
      }
    } catch (error) {
      console.error("API connection error:", error);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex h-screen bg-discord-main text-discord-text">
      {/* Sidebar - Lista Sesji */}
      <div className="w-64 bg-discord-sidebar flex flex-col hidden md:flex">
        <div className="p-4 shadow-md font-bold border-b border-discord-darker">Conversations</div>
        <div className="p-2 space-y-1 overflow-y-auto">
          <div className="bg-discord-input p-2 rounded text-white cursor-pointer">  Main session</div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="h-12 border-b border-discord-darker flex items-center px-4">
          <span className="font-bold text-white">UniLM Agent</span>
        </div>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map((msg) => (
            <div key={msg.id} className="flex items-start space-x-4 hover:bg-[#2e3035] p-1 rounded group">
              <div className={`w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center font-bold text-white ${msg.role === 'user' ? 'bg-indigo-500' : 'bg-emerald-500'}`}>
                {msg.role === 'user' ? 'U' : 'A'}
              </div>
              <div className="flex-1 overflow-hidden">
                <div className="flex items-center space-x-2">
                  <span className={`font-medium ${msg.role === 'user' ? 'text-indigo-400' : 'text-emerald-400'}`}>
                    {msg.role === 'user' ? 'Student' : 'AI Orchestrator'}
                  </span>
                </div>

                <div className="text-discord-text mt-1 leading-relaxed prose prose-invert max-w-none">
                  <ReactMarkdown
                    components={{
                      // Ensure paragraphs have consistent spacing
                      p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
                      code({ node, inline, className, children, ...props }: any) {
                        const match = /language-(\w+)/.exec(className || '');
                        return !inline && match ? (
                          <div className="my-4 rounded-md overflow-hidden border border-[#1e1f22] shadow-2xl">
                            {/* Optional: Add a small label for the language */}
                            <div className="bg-[#1e1f22] px-4 py-1 text-xs text-gray-400 font-mono border-b border-white/5">
                              {match[1].toUpperCase()}
                            </div>
                            <SyntaxHighlighter
                              style={oneDark}
                              language={match[1]}
                              PreTag="div"
                              customStyle={{ margin: 0, padding: '1rem' }}
                              {...props}
                            >
                              {String(children).replace(/\n$/, '')}
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

                  {/* Typing Animation */}
                  {isTyping && msg.role === 'assistant' && msg.content === '' && (
                    <div className="flex items-center space-x-2 text-discord-text/50 italic py-2">
                      <span className="w-2 h-2 bg-discord-blurple rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                      <span className="w-2 h-2 bg-discord-blurple rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                      <span className="w-2 h-2 bg-discord-blurple rounded-full animate-bounce"></span>
                      <span className="ml-2 animate-pulse text-sm">Thinking...</span>
                    </div>
                  )}

                </div>
              </div>
            </div>
          ))}
          <div ref={scrollRef} />
        </div>

        

        {/* Input Bar */}
        <div className="p-4 bg-discord-main">
          <div className="bg-discord-input rounded-lg px-4 py-1 flex items-center shadow-md border-2 border-transparent focus-within:border-discord-blurple transition-all">
            <button className="text-[#B5BAC1] hover:text-white pr-3 text-2xl font-light">+</button>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Ask agent."
              className="bg-transparent flex-1 outline-none py-3 text-discord-text placeholder:text-[#6D6F78]"
            />
            <div className="flex space-x-3 ml-2 text-xl grayscale hover:grayscale-0 cursor-pointer transition-all">
              <span>{'>'}</span>
            </div>
          </div>
          <div className="text-[10px] p-1 text-[#949BA4] mt-1 uppercase font-bold tracking-wider">
            Press <span className="hover:underline cursor-help">Enter</span> to send
          </div>
        </div>
      </div>
    </div>
  );
}