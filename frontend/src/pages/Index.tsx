import { useRef, useEffect } from 'react';
import { Header } from '@/components/Header';
import { SubjectSelector } from '@/components/SubjectSelector';
import { MessageBubble } from '@/components/MessageBubble';
import { TypingIndicator } from '@/components/TypingIndicator';
import { ChatInput } from '@/components/ChatInput';
import { EmptyState } from '@/components/EmptyState';
import { Button } from '@/components/ui/button';
import { useChat } from '@/hooks/useChat';
import { Trash2 } from 'lucide-react';

const Index = () => {
  const {
    messages,
    isLoading,
    selectedSubject,
    setSelectedSubject,
    sendMessage,
    clearChat,
  } = useChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSuggestionClick = (example: string) => {
    sendMessage(example);
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Background decoration */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-1/4 top-0 h-96 w-96 rounded-full bg-primary/5 blur-3xl" />
        <div className="absolute -right-1/4 bottom-0 h-96 w-96 rounded-full bg-physics/5 blur-3xl" />
      </div>

      <Header />

      <main className="relative flex flex-1 flex-col">
        {/* Subject selector bar */}
        <div className="sticky top-16 z-40 border-b border-border/30 bg-background/60 backdrop-blur-xl">
          <div className="container flex items-center justify-between gap-4 px-4 py-3 md:px-6">
            <SubjectSelector
              selected={selectedSubject}
              onChange={setSelectedSubject}
            />
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearChat}
                className="text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Clear
              </Button>
            )}
          </div>
        </div>

        {/* Chat area */}
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          {messages.length === 0 ? (
            <EmptyState onSuggestionClick={handleSuggestionClick} />
          ) : (
            <div className="container space-y-6 px-4 py-6 md:px-6">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              {isLoading && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="sticky bottom-0 border-t border-border/30 bg-background/80 backdrop-blur-xl">
          <div className="container px-4 py-4 md:px-6">
            <ChatInput onSend={sendMessage} isLoading={isLoading} />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
