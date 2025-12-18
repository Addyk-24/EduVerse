import { Bot } from 'lucide-react';

export function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl gradient-primary text-primary-foreground">
        <Bot className="h-4 w-4" />
      </div>

      <div className="card-glass rounded-2xl rounded-bl-md px-4 py-3">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-primary animate-typing" style={{ animationDelay: '0ms' }} />
          <span className="h-2 w-2 rounded-full bg-primary animate-typing" style={{ animationDelay: '200ms' }} />
          <span className="h-2 w-2 rounded-full bg-primary animate-typing" style={{ animationDelay: '400ms' }} />
        </div>
      </div>
    </div>
  );
}
