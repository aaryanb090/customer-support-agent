// src/components/ChatWindow.jsx
import React, { useState, useEffect, useRef } from "react";

export default function ChatWindow() {
  const [messages, setMessages] = useState([
    { from: "agent", text: "Hello there! 👋 How can I assist you today?" },
  ]);
  const [text, setText] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const messagesEnd = useRef(null);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setMessages((m) => [...m, { from: "user", text: trimmed }]);
    setText("");
    try {
      const resp = await fetch("http://127.0.0.1:8000/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "u1", message: trimmed, top_k: 3 }),
      });
      if (!resp.ok) throw new Error();
      const data = await resp.json();
      setMessages((m) => [...m, { from: "agent", text: data.answer }]);
    } catch {
      setMessages((m) => [
        ...m,
        { from: "agent", text: "⚠️ Oops, something went wrong." },
      ]);
    }
  };

  return (
    <div className={darkMode ? "app-container dark" : "app-container light"}>
      <div className="chat-container">
        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`bubble ${m.from}`}>
              {m.text}
            </div>
          ))}
          <div ref={messagesEnd} />
        </div>
        <div className="input-area">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Type a message…"
          />
          <button onClick={send}>Send</button>
        </div>
        <div className="toggle-area">
          <button onClick={() => setDarkMode((d) => !d)} className="toggle-btn">
            {darkMode ? "🌙 Dark" : "☀️ Light"}
          </button>
        </div>
      </div>
    </div>
  );
}
