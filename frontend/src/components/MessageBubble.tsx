import { Message } from '@/types/chat';
import { cn } from '@/lib/utils';
import { User, Bot, Atom, Calculator, FlaskConical, Leaf, Sparkles } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
}

const subjectIcons = {
  physics: <Atom className="h-3.5 w-3.5" />,
  maths: <Calculator className="h-3.5 w-3.5" />,
  chemistry: <FlaskConical className="h-3.5 w-3.5" />,
  biology: <Leaf className="h-3.5 w-3.5" />,
  general: <Sparkles className="h-3.5 w-3.5" />,
};

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={cn(
        "flex gap-3 animate-fade-in",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <div
        className={cn(
          "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl",
          isUser
            ? "bg-primary/20 text-primary"
            : "gradient-primary text-primary-foreground"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      <div
        className={cn(
          "flex max-w-[80%] flex-col gap-2",
          isUser ? "items-end" : "items-start"
        )}
      >
        {message.subject && isUser && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            {subjectIcons[message.subject]}
            <span className="capitalize">{message.subject}</span>
          </div>
        )}

        <div
          className={cn(
            "rounded-2xl px-4 py-3",
            isUser
              ? "bg-primary text-primary-foreground rounded-br-md"
              : "card-glass rounded-bl-md"
          )}
        >
          {message.image && (
            <div className="mb-3 overflow-hidden rounded-lg">
              <img
                src={message.image}
                alt="Uploaded content"
                className="max-h-48 w-auto object-contain"
              />
            </div>
          )}
          <p className="whitespace-pre-wrap text-sm leading-relaxed">
            {message.content}
          </p>
        </div>

        <span className="text-xs text-muted-foreground/60">
          {message.timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}
