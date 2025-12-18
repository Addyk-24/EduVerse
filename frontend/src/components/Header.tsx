import { Wifi, WifiOff, Brain } from 'lucide-react';
import { useEffect, useState } from 'react';

export function Header() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="container flex h-16 items-center justify-between px-4 md:px-6">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl gradient-primary">
              <Brain className="h-5 w-5 text-primary-foreground" />
            </div>
            <div className="absolute -bottom-1 -right-1 h-3 w-3 rounded-full border-2 border-background bg-primary animate-pulse" />
          </div>
          <div>
            <h1 className="font-display text-lg font-semibold tracking-tight">
              <span className="text-gradient">Gemma</span>
              <span className="text-foreground">3n</span>
            </h1>
            <p className="text-xs text-muted-foreground">Academic AI Assistant</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
            isOnline 
              ? 'bg-primary/10 text-primary' 
              : 'bg-destructive/10 text-destructive'
          }`}>
            {isOnline ? (
              <>
                <Wifi className="h-3.5 w-3.5" />
                <span>Online</span>
              </>
            ) : (
              <>
                <WifiOff className="h-3.5 w-3.5" />
                <span>Offline Mode</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
