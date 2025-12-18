export type Subject = 'physics' | 'maths' | 'chemistry' | 'biology' | 'general';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  image?: string;
  subject?: Subject;
  timestamp: Date;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  selectedSubject: Subject;
}
