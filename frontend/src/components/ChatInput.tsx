import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Send, ImagePlus, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSend: (message: string, image?: string) => void;
  isLoading: boolean;
}

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [image, setImage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() || image) {
      onSend(message.trim(), image || undefined);
      setMessage('');
      setImage(null);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      {image && (
        <div className="mb-3 flex items-start gap-2 rounded-xl border border-border/50 bg-secondary/30 p-3">
          <div className="relative">
            <img
              src={image}
              alt="Upload preview"
              className="h-20 w-auto rounded-lg object-cover"
            />
            <button
              type="button"
              onClick={() => setImage(null)}
              className="absolute -right-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-destructive text-destructive-foreground transition-transform hover:scale-110"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <span className="text-xs text-muted-foreground">Image attached</span>
        </div>
      )}

      <div className="flex items-end gap-3 rounded-2xl border border-border/50 bg-card/50 p-2 backdrop-blur-sm transition-all focus-within:border-primary/50 focus-within:shadow-[0_0_20px_hsl(174_72%_46%_/_0.1)]">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="shrink-0 text-muted-foreground hover:text-primary"
          onClick={() => fileInputRef.current?.click()}
        >
          <ImagePlus className="h-5 w-5" />
        </Button>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="hidden"
        />

        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
            adjustTextareaHeight();
          }}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about physics, maths, chemistry..."
          className={cn(
            "flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none",
            "min-h-[40px] max-h-[150px] py-2"
          )}
          rows={1}
        />

        <Button
          type="submit"
          variant="gradient"
          size="icon"
          disabled={isLoading || (!message.trim() && !image)}
          className={cn(
            "shrink-0 transition-all",
            isLoading && "animate-pulse"
          )}
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>

      <p className="mt-2 text-center text-xs text-muted-foreground/60">
        Press Enter to send, Shift + Enter for new line
      </p>
    </form>
  );
}
