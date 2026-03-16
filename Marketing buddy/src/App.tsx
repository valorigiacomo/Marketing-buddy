import React, { useState, useEffect, useRef } from 'react';
import { 
  Lightbulb, 
  TrendingUp, 
  MapPin, 
  DollarSign, 
  MessageSquare, 
  Plus, 
  Trash2, 
  ChevronRight, 
  Loader2,
  CheckCircle2,
  AlertCircle,
  BarChart3,
  Sparkles,
  FileText,
  Send,
  ArrowRight,
  Wallet,
  Globe,
  Users,
  Search,
  Save,
  FolderOpen,
  Zap,
  Upload,
  PieChart as PieChartIcon,
  LineChart as LineChartIcon,
  HelpCircle,
  Info,
  Activity,
  Scale,
  Briefcase
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell,
  PieChart,
  Pie,
  LineChart,
  Line,
  Legend,
  AreaChart,
  Area
} from 'recharts';
import * as XLSX from 'xlsx';
import Markdown from 'react-markdown';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from './lib/utils';
import { 
  analyzeMarketingIdea, 
  analyzeLocation, 
  findSponsorsAI,
  createMarketingChat, 
  createAccountantChat,
  MarketingAnalysis 
} from './services/geminiService';

// --- Types ---

interface FinancialItem {
  id: string;
  description: string;
  amount: number;
  type: 'income' | 'expense';
  category: string;
  date: string; // ISO format YYYY-MM-DD
  isRecurring?: boolean;
}

interface CategoryBudget {
  category: string;
  limit: number;
}

interface Message {
  role: 'user' | 'model';
  text: string;
}

interface SavedIdea {
  id: string;
  name: string;
  timestamp: number;
  idea: string;
  projectCategory: string;
  location: string;
  radius: number;
  businessType: string;
  analysis: MarketingAnalysis | null;
  financialItems: FinancialItem[];
  locationAnalysis: { text: string; sources: any[] } | null;
  sponsorAI: string | null;
}

// --- Components ---

export default function App() {
  const [idea, setIdea] = useState('');
  const [projectCategory, setProjectCategory] = useState('evento');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<MarketingAnalysis | null>(null);
  const [activeTab, setActiveTab] = useState<'analysis' | 'financials' | 'location' | 'chat'>('analysis');
  
  // Financials State
  const [financialItems, setFinancialItems] = useState<FinancialItem[]>([]);
  const [newItemDesc, setNewItemDesc] = useState('');
  const [newItemAmount, setNewItemAmount] = useState('');
  const [newItemType, setNewItemType] = useState<'income' | 'expense'>('expense');
  const [newItemCategory, setNewItemCategory] = useState('Marketing');
  const [newItemDate, setNewItemDate] = useState(new Date().toISOString().split('T')[0]);
  const [newItemIsRecurring, setNewItemIsRecurring] = useState(false);
  const [chartType, setChartType] = useState<'bar' | 'line' | 'pie' | 'area'>('bar');
  const [projectionMonths, setProjectionMonths] = useState(6);
  const [categoryBudgets, setCategoryBudgets] = useState<CategoryBudget[]>([
    { category: 'Marketing', limit: 1000 },
    { category: 'Operativo', limit: 500 },
    { category: 'Personale', limit: 2000 },
    { category: 'Software', limit: 200 },
    { category: 'Altro', limit: 300 }
  ]);
  const [showBudgetModal, setShowBudgetModal] = useState(false);

  // Location State
  const [location, setLocation] = useState('');
  const [radius, setRadius] = useState(5);
  const [businessType, setBusinessType] = useState('all');
  const [isLocAnalyzing, setIsLocAnalyzing] = useState(false);
  const [locationAnalysis, setLocationAnalysis] = useState<{ text: string; sources: any[] } | null>(null);
  const [sponsorAI, setSponsorAI] = useState<string | null>(null);
  const [isSponsorAIAnalyzing, setIsSponsorAIAnalyzing] = useState(false);

  // Chat State
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [chatPersona, setChatPersona] = useState<'marketing' | 'accountant'>('marketing');
  const [savedIdeas, setSavedIdeas] = useState<SavedIdea[]>([]);
  const [showSavedList, setShowSavedList] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const chatInstance = useRef<any>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  useEffect(() => {
    const stored = localStorage.getItem('marketing_buddy_ideas');
    if (stored) {
      try {
        setSavedIdeas(JSON.parse(stored));
      } catch (e) {
        console.error("Failed to parse saved ideas", e);
      }
    }
  }, []);

  const saveToLocalStorage = (ideas: SavedIdea[]) => {
    localStorage.setItem('marketing_buddy_ideas', JSON.stringify(ideas));
    setSavedIdeas(ideas);
  };

  const handleSaveIdea = () => {
    if (!idea.trim() || !saveName.trim()) return;
    
    const newIdea: SavedIdea = {
      id: Math.random().toString(36).substr(2, 9),
      name: saveName,
      timestamp: Date.now(),
      idea,
      projectCategory,
      location,
      radius,
      businessType,
      analysis,
      financialItems,
      locationAnalysis,
      sponsorAI
    };

    saveToLocalStorage([newIdea, ...savedIdeas]);
    setShowSaveModal(false);
    setSaveName('');
    setNotification({ message: "Progetto salvato con successo!", type: 'success' });
  };

  const handleLoadIdea = (saved: SavedIdea) => {
    setIdea(saved.idea);
    setProjectCategory(saved.projectCategory || 'evento');
    setLocation(saved.location);
    setRadius(saved.radius);
    setBusinessType(saved.businessType);
    setAnalysis(saved.analysis);
    setFinancialItems(saved.financialItems);
    setLocationAnalysis(saved.locationAnalysis);
    setSponsorAI(saved.sponsorAI);
    setShowSavedList(false);
    setActiveTab('analysis');
    setNotification({ message: "Progetto caricato!", type: 'success' });
  };

  const confirmDelete = (id: string) => {
    saveToLocalStorage(savedIdeas.filter(i => i.id !== id));
    setDeleteConfirm(null);
    setNotification({ message: "Progetto eliminato.", type: 'success' });
  };

  useEffect(() => {
    if (chatPersona === 'marketing') {
      chatInstance.current = createMarketingChat();
    } else {
      chatInstance.current = createAccountantChat();
    }
    setChatMessages([]);
  }, [chatPersona]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleAnalyze = async () => {
    if (!idea.trim()) return;
    setIsAnalyzing(true);
    try {
      const result = await analyzeMarketingIdea(idea, projectCategory);
      setAnalysis(result);
      setActiveTab('analysis');
    } catch (error) {
      console.error(error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleLocationAnalyze = async () => {
    if (!location.trim() || !idea.trim()) return;
    setIsLocAnalyzing(true);
    setIsSponsorAIAnalyzing(true);
    try {
      let lat, lng;
      try {
        // Aggiungiamo un timeout alla geolocalizzazione per evitare blocchi infiniti
        const pos = await new Promise<GeolocationPosition>((res, rej) => {
          const timeoutId = setTimeout(() => rej(new Error("Timeout geolocalizzazione")), 5000);
          navigator.geolocation.getCurrentPosition(
            (position) => {
              clearTimeout(timeoutId);
              res(position);
            },
            (error) => {
              clearTimeout(timeoutId);
              rej(error);
            },
            { timeout: 5000 }
          );
        });
        lat = pos.coords.latitude;
        lng = pos.coords.longitude;
      } catch (e) {
        console.log("Geolocalizzazione non disponibile o timeout:", e);
      }

      const [locResult, aiSponsors] = await Promise.all([
        analyzeLocation(location, idea, projectCategory, radius, businessType, lat, lng).catch(err => ({ 
          text: `Errore Mappe: ${err.message}`, 
          sources: [] 
        })),
        findSponsorsAI(idea, projectCategory, location).catch(err => `Errore AI: ${err.message}`)
      ]);
      
      setLocationAnalysis(locResult);
      setSponsorAI(aiSponsors);
    } catch (error: any) {
      console.error("Errore fatale durante l'analisi della località:", error);
      setLocationAnalysis({ text: `Errore critico: ${error.message}`, sources: [] });
      setSponsorAI("Si è verificato un errore imprevisto.");
    } finally {
      setIsLocAnalyzing(false);
      setIsSponsorAIAnalyzing(false);
    }
  };

  const handleAddFinancial = () => {
    if (!newItemDesc || !newItemAmount) return;
    const item: FinancialItem = {
      id: Math.random().toString(36).substr(2, 9),
      description: newItemDesc,
      amount: parseFloat(newItemAmount),
      type: newItemType,
      category: newItemCategory,
      date: newItemDate,
      isRecurring: newItemIsRecurring
    };
    setFinancialItems([...financialItems, item]);
    setNewItemDesc('');
    setNewItemAmount('');
    setNewItemIsRecurring(false);
  };

  const handleExportExcel = () => {
    const ws = XLSX.utils.json_to_sheet(financialItems.map(item => ({
      Descrizione: item.description,
      Importo: item.amount,
      Tipo: item.type === 'income' ? 'Entrata' : 'Uscita',
      Categoria: item.category,
      Data: item.date,
      Ricorrente: item.isRecurring ? 'Sì' : 'No'
    })));
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Budget");
    XLSX.writeFile(wb, "Marketing_Budget_Export.xlsx");
  };

  const updateBudget = (category: string, limit: number) => {
    setCategoryBudgets(prev => prev.map(b => b.category === category ? { ...b, limit } : b));
  };

  const getFinancialHealth = () => {
    const monthlyExpenses = financialItems
      .filter(i => i.type === 'expense')
      .reduce((acc, curr) => acc + curr.amount, 0);
    
    // Average monthly burn (simplified)
    const months = new Set(financialItems.map(i => i.date.substring(0, 7))).size || 1;
    const burnRate = monthlyExpenses / months;
    
    const currentCash = financialItems.reduce((acc, curr) => 
      curr.type === 'income' ? acc + curr.amount : acc - curr.amount, 0);
    
    const runway = burnRate > 0 ? Math.max(0, currentCash / burnRate) : Infinity;

    return { burnRate, runway, currentCash };
  };

  const health = getFinancialHealth();

  const getHealthStatus = () => {
    if (health.runway === Infinity) return { label: 'Eccellente', color: 'text-emerald-600', desc: 'Le tue entrate coprono tutte le spese. Situazione ideale.' };
    if (health.runway > 12) return { label: 'Stabile', color: 'text-emerald-500', desc: 'Hai oltre un anno di autonomia. Puoi pianificare investimenti.' };
    if (health.runway > 6) return { label: 'Buona', color: 'text-indigo-500', desc: 'Situazione sotto controllo, ma monitora i costi fissi.' };
    if (health.runway > 3) return { label: 'Attenzione', color: 'text-amber-500', desc: 'Autonomia limitata. Cerca di aumentare le entrate o tagliare i costi.' };
    return { label: 'Critica', color: 'text-rose-600', desc: 'Autonomia inferiore a 3 mesi. Intervento urgente necessario.' };
  };

  const status = getHealthStatus();

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      const bstr = evt.target?.result;
      const wb = XLSX.read(bstr, { type: 'binary' });
      const wsname = wb.SheetNames[0];
      const ws = wb.Sheets[wsname];
      const data = XLSX.utils.sheet_to_json(ws) as any[];

      const newItems: FinancialItem[] = data.map((row) => ({
        id: Math.random().toString(36).substr(2, 9),
        description: row.Descrizione || row.description || 'Importato',
        amount: parseFloat(row.Importo || row.amount || 0),
        type: (row.Tipo || row.type || 'expense').toLowerCase().includes('entrata') || (row.Tipo || row.type || '').toLowerCase().includes('income') ? 'income' : 'expense',
        category: row.Categoria || row.category || 'Altro',
        date: row.Data || row.date || new Date().toISOString().split('T')[0]
      }));

      setFinancialItems(prev => [...prev, ...newItems]);
      setNotification({ message: `${newItems.length} voci importate con successo!`, type: 'success' });
    };
    reader.readAsBinaryString(file);
  };

  const getProjectionData = () => {
    if (financialItems.length === 0) return [];

    const monthlyData: { [key: string]: { income: number, expense: number } } = {};
    financialItems.forEach(item => {
      const month = item.date.substring(0, 7); // YYYY-MM
      if (!monthlyData[month]) monthlyData[month] = { income: 0, expense: 0 };
      if (item.type === 'income') monthlyData[month].income += item.amount;
      else monthlyData[month].expense += item.amount;
    });

    const sortedMonths = Object.keys(monthlyData).sort();
    const result = sortedMonths.map(month => ({
      name: month,
      income: monthlyData[month].income,
      expense: monthlyData[month].expense,
      profit: monthlyData[month].income - monthlyData[month].expense
    }));

    // Simple Linear Projection
    if (result.length >= 1) {
      const lastMonth = result[result.length - 1];
      const avgGrowthIncome = result.length > 1 ? (result[result.length - 1].income - result[0].income) / (result.length - 1) : 0;
      const avgGrowthExpense = result.length > 1 ? (result[result.length - 1].expense - result[0].expense) / (result.length - 1) : 0;

      let currentMonth = new Date(lastMonth.name + '-02'); // Use 02 to avoid timezone issues
      for (let i = 1; i <= projectionMonths; i++) {
        currentMonth.setMonth(currentMonth.getMonth() + 1);
        const monthStr = currentMonth.toISOString().substring(0, 7);
        const projectedIncome = Math.max(0, lastMonth.income + (avgGrowthIncome * i));
        const projectedExpense = Math.max(0, lastMonth.expense + (avgGrowthExpense * i));
        
        result.push({
          name: monthStr + ' (Proiezione)',
          income: Math.round(projectedIncome),
          expense: Math.round(projectedExpense),
          profit: Math.round(projectedIncome - projectedExpense)
        });
      }
    }

    return result;
  };

  const getPieData = () => {
    const categories: { [key: string]: number } = {};
    financialItems.forEach(item => {
      if (item.type === 'expense') {
        categories[item.category] = (categories[item.category] || 0) + item.amount;
      }
    });
    return Object.keys(categories).map(cat => ({ name: cat, value: categories[cat] }));
  };

  const removeFinancial = (id: string) => {
    setFinancialItems(financialItems.filter(i => i.id !== id));
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim() || isChatLoading) return;
    
    const userMsg: Message = { role: 'user', text: chatInput };
    setChatMessages(prev => [...prev, userMsg]);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const response = await chatInstance.current.sendMessage({ message: chatInput });
      const modelMsg: Message = { role: 'model', text: response.text };
      setChatMessages(prev => [...prev, modelMsg]);
    } catch (error) {
      console.error(error);
    } finally {
      setIsChatLoading(false);
    }
  };

  const totalIncome = financialItems.filter(i => i.type === 'income').reduce((acc, i) => acc + i.amount, 0);
  const totalExpense = financialItems.filter(i => i.type === 'expense').reduce((acc, i) => acc + i.amount, 0);
  const netProfit = totalIncome - totalExpense;

  const chartData = [
    { name: 'Entrate', value: totalIncome, fill: '#6366f1' },
    { name: 'Uscite', value: totalExpense, fill: '#f43f5e' }
  ];

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      {/* Modern Navbar */}
      <nav className="sticky top-0 z-50 bg-white/70 backdrop-blur-xl border-b border-slate-200/60">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-200 rotate-3">
              <Sparkles className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-extrabold tracking-tight text-slate-900">Marketing <span className="text-indigo-600">buddy</span> AI</h1>
              <p className="text-[10px] uppercase tracking-widest font-bold text-slate-400">Il Tuo Compagno Strategico</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-1 bg-slate-100 p-1 rounded-xl border border-slate-200/50">
            {[
              { id: 'analysis', label: 'Analisi', icon: FileText },
              { id: 'financials', label: 'Finanza', icon: Wallet },
              { id: 'location', label: 'Località', icon: Globe },
              { id: 'chat', label: 'Chat', icon: MessageSquare },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all",
                  activeTab === tab.id 
                    ? "bg-white text-indigo-600 shadow-sm" 
                    : "text-slate-500 hover:text-slate-900 hover:bg-white/50"
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-6 lg:p-10 grid grid-cols-1 lg:grid-cols-12 gap-10">
        {/* Left: Input Section */}
        <div className="lg:col-span-4 space-y-8">
          <section className="bg-white rounded-3xl p-8 shadow-xl shadow-slate-200/50 border border-slate-100">
            <div className="flex items-center gap-2 mb-6">
              <div className="w-8 h-8 bg-indigo-50 rounded-lg flex items-center justify-center">
                <Lightbulb className="w-4 h-4 text-indigo-600" />
              </div>
              <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">La Visione</h2>
            </div>
            
            <textarea
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              placeholder="Qual è la tua prossima grande mossa di marketing? Descrivila qui..."
              className="w-full h-40 p-5 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 outline-none transition-all resize-none text-slate-700 placeholder:text-slate-400"
            />

            <div className="mt-4 space-y-2">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest ml-1">Categoria Progetto</label>
              <select
                value={projectCategory}
                onChange={(e) => setProjectCategory(e.target.value)}
                className="w-full p-4 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 outline-none transition-all appearance-none text-sm font-semibold text-slate-700"
              >
                <option value="evento">Evento (Musica, Sport, Sociale)</option>
                <option value="attività professionale">Attività Professionale (Studio, Consulenza)</option>
                <option value="corso">Corso / Formazione</option>
                <option value="prodotto">Lancio Prodotto</option>
                <option value="locale">Attività Locale (Negozio, Ristorante)</option>
              </select>
            </div>
            
            <div className="grid grid-cols-2 gap-3 mt-4">
              <button
                onClick={() => {
                  setSaveName(idea.substring(0, 20) + "...");
                  setShowSaveModal(true);
                }}
                disabled={!idea.trim()}
                className="flex items-center justify-center gap-2 py-3 rounded-xl border border-slate-200 text-slate-600 font-bold text-xs hover:bg-slate-50 transition-all disabled:opacity-50"
              >
                <Save className="w-4 h-4" /> Salva
              </button>
              <button
                onClick={() => setShowSavedList(!showSavedList)}
                className="flex items-center justify-center gap-2 py-3 rounded-xl border border-slate-200 text-slate-600 font-bold text-xs hover:bg-slate-50 transition-all"
              >
                <FolderOpen className="w-4 h-4" /> Carica
              </button>
            </div>

            <AnimatePresence>
              {showSaveModal && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="mt-4 p-4 bg-indigo-50 rounded-2xl border border-indigo-100 space-y-3"
                >
                  <label className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">Nome Progetto</label>
                  <input
                    type="text"
                    value={saveName}
                    onChange={(e) => setSaveName(e.target.value)}
                    className="w-full p-3 rounded-xl bg-white border border-indigo-200 focus:border-indigo-500 outline-none text-sm"
                    autoFocus
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={handleSaveIdea}
                      className="flex-1 bg-indigo-600 text-white py-2 rounded-xl text-xs font-bold hover:bg-indigo-700 transition-all"
                    >
                      Conferma
                    </button>
                    <button
                      onClick={() => setShowSaveModal(false)}
                      className="flex-1 bg-white text-slate-600 py-2 rounded-xl text-xs font-bold border border-slate-200 hover:bg-slate-50 transition-all"
                    >
                      Annulla
                    </button>
                  </div>
                </motion.div>
              )}

              {showSavedList && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-4 overflow-hidden"
                >
                  <div className="bg-slate-50 rounded-2xl border border-slate-200 p-2 space-y-1 max-h-60 overflow-y-auto custom-scrollbar">
                    {savedIdeas.length === 0 ? (
                      <p className="text-[10px] text-slate-400 text-center py-4 font-bold uppercase tracking-widest">Nessun progetto salvato</p>
                    ) : (
                      savedIdeas.map((saved) => (
                        <div
                          key={saved.id}
                          onClick={() => handleLoadIdea(saved)}
                          className="flex items-center justify-between p-3 rounded-xl hover:bg-white hover:shadow-sm cursor-pointer transition-all group"
                        >
                          <div className="flex flex-col">
                            <span className="text-xs font-bold text-slate-700 truncate max-w-[150px]">{saved.name}</span>
                            <span className="text-[9px] text-slate-400 font-medium">{new Date(saved.timestamp).toLocaleDateString()}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            {deleteConfirm === saved.id ? (
                              <div className="flex items-center gap-1">
                                <button
                                  onClick={(e) => { e.stopPropagation(); confirmDelete(saved.id); }}
                                  className="p-1.5 bg-rose-500 text-white rounded-lg hover:bg-rose-600 transition-colors"
                                >
                                  <CheckCircle2 className="w-3 h-3" />
                                </button>
                                <button
                                  onClick={(e) => { e.stopPropagation(); setDeleteConfirm(null); }}
                                  className="p-1.5 bg-slate-200 text-slate-600 rounded-lg hover:bg-slate-300 transition-colors"
                                >
                                  <Trash2 className="w-3 h-3" />
                                </button>
                              </div>
                            ) : (
                              <button
                                onClick={(e) => { e.stopPropagation(); setDeleteConfirm(saved.id); }}
                                className="p-2 text-slate-300 hover:text-rose-500 transition-colors"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing || !idea.trim()}
              className="w-full mt-6 bg-indigo-600 text-white py-4 rounded-2xl font-bold hover:bg-indigo-700 transition-all shadow-lg shadow-indigo-200 flex items-center justify-center gap-3 disabled:opacity-50 disabled:shadow-none group"
            >
              {isAnalyzing ? (
                <Loader2 className="animate-spin w-5 h-5" />
              ) : (
                <>
                  <span>Genera Strategia</span>
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </>
              )}
            </button>
          </section>

          {/* Mobile Nav */}
          <div className="md:hidden grid grid-cols-2 gap-3">
            {[
              { id: 'analysis', label: 'Strategia', icon: FileText },
              { id: 'financials', label: 'Finanza', icon: Wallet },
              { id: 'location', label: 'Mappa', icon: Globe },
              { id: 'chat', label: 'Chat', icon: MessageSquare },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  "flex items-center gap-2 p-4 rounded-2xl text-xs font-bold transition-all border",
                  activeTab === tab.id 
                    ? "bg-indigo-600 text-white border-indigo-600 shadow-lg shadow-indigo-100" 
                    : "bg-white text-slate-600 border-slate-100 hover:border-indigo-200"
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Right: Content Area */}
        <div className="lg:col-span-8">
          <AnimatePresence mode="wait">
            {activeTab === 'analysis' && (
              <motion.div
                key="analysis"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-8"
              >
                {!analysis ? (
                  <div className="h-[500px] bg-white rounded-[40px] border border-slate-100 flex flex-col items-center justify-center text-center p-12">
                    <div className="w-20 h-20 bg-indigo-50 rounded-full flex items-center justify-center mb-6">
                      <Sparkles className="w-10 h-10 text-indigo-400" />
                    </div>
                    <h3 className="text-xl font-bold text-slate-900 mb-2">Pronto a pianificare?</h3>
                    <p className="text-slate-500 max-w-xs mx-auto">Inserisci la tua idea a sinistra per sbloccare un'analisi completa basata su AI.</p>
                  </div>
                ) : (
                  <>
                    {/* SWOT Bento Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {[
                        { title: 'Punti di Forza', items: analysis.swot.strengths, icon: CheckCircle2, color: 'text-emerald-600', bg: 'bg-emerald-50/50' },
                        { title: 'Debolezze', items: analysis.swot.weaknesses, icon: AlertCircle, color: 'text-amber-600', bg: 'bg-amber-50/50' },
                        { title: 'Opportunità', items: analysis.swot.opportunities, icon: TrendingUp, color: 'text-indigo-600', bg: 'bg-indigo-50/50' },
                        { title: 'Minacce', items: analysis.swot.threats, icon: AlertCircle, color: 'text-rose-600', bg: 'bg-rose-50/50' },
                      ].map((section) => (
                        <div key={section.title} className={cn("p-8 rounded-[32px] border border-slate-100 bg-white shadow-sm", section.bg)}>
                          <div className="flex items-center gap-3 mb-6">
                            <section.icon className={cn("w-5 h-5", section.color)} />
                            <h3 className="font-bold text-slate-900 tracking-tight">{section.title}</h3>
                          </div>
                          <ul className="space-y-3">
                            {section.items.map((item, i) => (
                              <li key={i} className="flex gap-3 text-sm text-slate-600 leading-relaxed">
                                <span className={cn("w-1.5 h-1.5 rounded-full mt-1.5 shrink-0", section.color.replace('text', 'bg'))} />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>

                    {/* Business Plan Card */}
                    <div className="bg-white rounded-[40px] p-10 border border-slate-100 shadow-sm">
                      <div className="flex items-center gap-3 mb-8 pb-6 border-b border-slate-50">
                        <FileText className="w-6 h-6 text-indigo-600" />
                        <h3 className="text-2xl font-extrabold text-slate-900 tracking-tight">Roadmap Strategica</h3>
                      </div>
                      <div className="prose prose-slate max-w-none prose-headings:text-slate-900 prose-headings:font-extrabold prose-p:text-slate-600 prose-li:text-slate-600">
                        <Markdown>{analysis.businessPlan}</Markdown>
                      </div>
                    </div>
                  </>
                )}
              </motion.div>
            )}

            {activeTab === 'financials' && (
              <motion.div
                key="financials"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-8"
              >
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Modern Input Form */}
                    <div className="bg-white rounded-[32px] p-8 border border-slate-100 shadow-sm">
                      <div className="flex items-center justify-between mb-8">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Nuova Voce</h3>
                        <div className="flex gap-2">
                          <button 
                            onClick={handleExportExcel}
                            className="flex items-center gap-2 px-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-[10px] font-bold text-slate-600 hover:bg-slate-100 transition-all"
                          >
                            <Save className="w-3 h-3" /> Esporta
                          </button>
                          <label className="flex items-center gap-2 px-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-[10px] font-bold text-slate-600 cursor-pointer hover:bg-slate-100 transition-all">
                            <Upload className="w-3 h-3" />
                            Importa
                            <input type="file" accept=".xlsx, .xls, .csv" onChange={handleFileUpload} className="hidden" />
                          </label>
                        </div>
                      </div>
                      <div className="space-y-6">
                        <div className="space-y-2">
                          <label className="text-xs font-bold text-slate-500 ml-1">Descrizione</label>
                          <input
                            type="text"
                            value={newItemDesc}
                            onChange={(e) => setNewItemDesc(e.target.value)}
                            className="w-full p-4 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 outline-none transition-all"
                            placeholder="es. Campagna Influencer"
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <label className="text-xs font-bold text-slate-500 ml-1">Importo (€)</label>
                            <input
                              type="number"
                              value={newItemAmount}
                              onChange={(e) => setNewItemAmount(e.target.value)}
                              className="w-full p-4 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 outline-none transition-all"
                              placeholder="0.00"
                            />
                          </div>
                          <div className="space-y-2">
                            <label className="text-xs font-bold text-slate-500 ml-1">Tipo</label>
                            <select
                              value={newItemType}
                              onChange={(e) => setNewItemType(e.target.value as any)}
                              className="w-full p-4 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 outline-none transition-all appearance-none"
                            >
                              <option value="expense">Uscita</option>
                              <option value="income">Entrata</option>
                            </select>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <label className="text-xs font-bold text-slate-500 ml-1">Categoria</label>
                            <select
                              value={newItemCategory}
                              onChange={(e) => setNewItemCategory(e.target.value)}
                              className="w-full p-4 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 outline-none transition-all appearance-none"
                            >
                              <option value="Marketing">Marketing & Pubblicità</option>
                              <option value="Operativo">Costi Operativi</option>
                              <option value="Personale">Personale / Collaboratori</option>
                              <option value="Software">Software & Tool</option>
                              <option value="Altro">Altro</option>
                            </select>
                          </div>
                          <div className="space-y-2">
                            <label className="text-xs font-bold text-slate-500 ml-1">Data</label>
                            <input
                              type="date"
                              value={newItemDate}
                              onChange={(e) => setNewItemDate(e.target.value)}
                              className="w-full p-4 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 outline-none transition-all"
                            />
                          </div>
                        </div>
                        <div className="flex items-center gap-3 ml-1">
                          <input
                            type="checkbox"
                            id="recurring"
                            checked={newItemIsRecurring}
                            onChange={(e) => setNewItemIsRecurring(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                          />
                          <label htmlFor="recurring" className="text-xs font-bold text-slate-500 cursor-pointer">Voce Ricorrente (Mensile)</label>
                        </div>
                        <button
                          onClick={handleAddFinancial}
                          className="w-full bg-indigo-600 text-white py-4 rounded-2xl font-bold hover:bg-indigo-700 transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-100"
                        >
                          <Plus className="w-5 h-5" /> Aggiungi al Budget
                        </button>
                      </div>
                    </div>

                    {/* Visual Stats & List */}
                    <div className="space-y-6">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-white rounded-[32px] p-6 border border-slate-100 shadow-sm">
                          <div className="w-8 h-8 bg-emerald-50 rounded-xl flex items-center justify-center mb-3">
                            <TrendingUp className="w-4 h-4 text-emerald-600" />
                          </div>
                          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Entrate</span>
                          <div className="text-xl font-extrabold text-slate-900 mt-1">€{totalIncome.toLocaleString()}</div>
                        </div>
                        <div className="bg-white rounded-[32px] p-6 border border-slate-100 shadow-sm">
                          <div className="w-8 h-8 bg-rose-50 rounded-xl flex items-center justify-center mb-3">
                            <DollarSign className="w-4 h-4 text-rose-600" />
                          </div>
                          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Uscite</span>
                          <div className="text-xl font-extrabold text-slate-900 mt-1">€{totalExpense.toLocaleString()}</div>
                        </div>
                      </div>
                      
                      <div className="bg-indigo-600 rounded-[32px] p-6 text-white shadow-xl shadow-indigo-200 relative overflow-hidden">
                        <div className="relative z-10">
                          <div className="flex justify-between items-start">
                            <div>
                              <div className="flex items-center gap-1.5 opacity-80">
                                <span className="text-[10px] font-bold uppercase tracking-widest">Saldo Netto</span>
                                <div className="group relative">
                                  <HelpCircle className="w-3 h-3 cursor-help" />
                                  <div className="absolute bottom-full left-0 mb-2 w-48 p-2 bg-slate-900 text-[9px] text-white rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                                    Il denaro totale che hai in cassa dopo aver sottratto tutte le spese dalle entrate.
                                  </div>
                                </div>
                              </div>
                              <div className="text-3xl font-black mt-1 tracking-tighter">€{netProfit.toLocaleString()}</div>
                            </div>
                            <div className="text-right">
                              <div className="flex items-center justify-end gap-1.5 opacity-80">
                                <span className="text-[10px] font-bold uppercase tracking-widest">Autonomia</span>
                                <div className="group relative">
                                  <HelpCircle className="w-3 h-3 cursor-help" />
                                  <div className="absolute bottom-full right-0 mb-2 w-48 p-2 bg-slate-900 text-[9px] text-white rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 text-right">
                                    Quanti mesi puoi sopravvivere con i fondi attuali se non avessi nuove entrate.
                                  </div>
                                </div>
                              </div>
                              <div className="text-xl font-bold mt-1">{health.runway === Infinity ? '∞' : health.runway.toFixed(1)} Mesi</div>
                            </div>
                          </div>
                          
                          <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-2 gap-4">
                            <div>
                              <div className="flex items-center gap-1.5 opacity-80">
                                <span className="text-[9px] font-bold uppercase">Burn Rate</span>
                                <div className="group relative">
                                  <HelpCircle className="w-2.5 h-2.5 cursor-help" />
                                  <div className="absolute bottom-full left-0 mb-2 w-48 p-2 bg-slate-900 text-[9px] text-white rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                                    La tua spesa media mensile. Indica quanto "bruci" ogni mese per operare.
                                  </div>
                                </div>
                              </div>
                              <div className="text-sm font-bold">€{health.burnRate.toLocaleString()}/mese</div>
                            </div>
                            <div className="text-right">
                              <div className="flex items-center justify-end gap-1.5 opacity-80">
                                <span className="text-[9px] font-bold uppercase">ROI</span>
                                <div className="group relative">
                                  <HelpCircle className="w-2.5 h-2.5 cursor-help" />
                                  <div className="absolute bottom-full right-0 mb-2 w-48 p-2 bg-slate-900 text-[9px] text-white rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 text-right">
                                    Ritorno sull'Investimento. Indica quanto guadagni per ogni euro speso.
                                  </div>
                                </div>
                              </div>
                              <div className="text-sm font-bold">{totalExpense > 0 ? ((netProfit / totalExpense) * 100).toFixed(1) : 0}%</div>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Verbal Health Summary */}
                      <div className="bg-white rounded-[32px] p-6 border border-slate-100 shadow-sm">
                        <div className="flex items-center gap-3 mb-3">
                          <div className={cn("w-2 h-2 rounded-full animate-pulse", status.color.replace('text', 'bg'))} />
                          <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Stato Finanziario</h4>
                        </div>
                        <div className={cn("text-lg font-black mb-1", status.color)}>{status.label}</div>
                        <p className="text-xs text-slate-500 leading-relaxed font-medium">{status.desc}</p>
                      </div>

                      {/* Budgeting Monitor */}
                      <div className="bg-white rounded-[32px] p-6 border border-slate-100 shadow-sm">
                        <div className="flex items-center justify-between mb-4">
                          <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Monitoraggio Budget</h4>
                          <button 
                            onClick={() => setShowBudgetModal(true)}
                            className="text-[10px] font-bold text-indigo-600 hover:underline"
                          >
                            Imposta Limiti
                          </button>
                        </div>
                        <div className="space-y-4">
                          {categoryBudgets.map((budget) => {
                            const spent = financialItems
                              .filter(i => i.category === budget.category && i.type === 'expense')
                              .reduce((acc, curr) => acc + curr.amount, 0);
                            const percent = Math.min(100, (spent / budget.limit) * 100);
                            
                            return (
                              <div key={budget.category} className="space-y-1.5">
                                <div className="flex justify-between text-[10px] font-bold">
                                  <span className="text-slate-600">{budget.category}</span>
                                  <span className={cn(percent > 90 ? "text-rose-600" : "text-slate-400")}>
                                    €{spent.toLocaleString()} / €{budget.limit.toLocaleString()}
                                  </span>
                                </div>
                                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                  <motion.div 
                                    initial={{ width: 0 }}
                                    animate={{ width: `${percent}%` }}
                                    className={cn(
                                      "h-full rounded-full",
                                      percent > 90 ? "bg-rose-500" : percent > 70 ? "bg-amber-500" : "bg-indigo-500"
                                    )}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Financial Items List */}
                      <div className="bg-white rounded-[32px] p-6 border border-slate-100 shadow-sm max-h-[300px] overflow-y-auto custom-scrollbar">
                        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4">Dettaglio Movimenti</h4>
                        <div className="space-y-3">
                          {financialItems.length === 0 ? (
                            <p className="text-center py-10 text-slate-300 text-xs font-bold italic">Nessun movimento registrato</p>
                          ) : (
                            financialItems.map((item) => (
                              <div key={item.id} className="flex items-center justify-between p-3 bg-slate-50 rounded-2xl group">
                                <div className="flex items-center gap-3">
                                  <div className={cn(
                                    "w-8 h-8 rounded-xl flex items-center justify-center",
                                    item.type === 'income' ? "bg-emerald-100 text-emerald-600" : "bg-rose-100 text-rose-600"
                                  )}>
                                    {item.type === 'income' ? <TrendingUp className="w-4 h-4" /> : <DollarSign className="w-4 h-4" />}
                                  </div>
                                  <div>
                                    <div className="flex items-center gap-2">
                                      <div className="text-xs font-bold text-slate-800">{item.description}</div>
                                      {item.isRecurring && (
                                        <div className="px-1.5 py-0.5 bg-indigo-50 text-indigo-600 text-[8px] font-bold rounded uppercase">Ricorrente</div>
                                      )}
                                    </div>
                                    <div className="text-[9px] font-bold text-slate-400 uppercase">{item.category} • {new Date(item.date).toLocaleDateString('it-IT')}</div>
                                  </div>
                                </div>
                                <div className="flex items-center gap-3">
                                  <span className={cn(
                                    "text-xs font-bold",
                                    item.type === 'income' ? "text-emerald-600" : "text-rose-600"
                                  )}>
                                    {item.type === 'income' ? '+' : '-'}€{item.amount.toLocaleString()}
                                  </span>
                                  <button 
                                    onClick={() => removeFinancial(item.id)}
                                    className="p-1.5 text-slate-300 hover:text-rose-500 opacity-0 group-hover:opacity-100 transition-all"
                                  >
                                    <Trash2 className="w-3.5 h-3.5" />
                                  </button>
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                {/* Chart Section */}
                <div className="bg-white rounded-[40px] p-10 border border-slate-100 shadow-sm">
                  <div className="flex flex-col lg:flex-row lg:items-center justify-between mb-10 gap-6">
                    <div>
                      <h3 className="text-xl font-extrabold text-slate-900 tracking-tight">Panoramica Finanziaria</h3>
                      <p className="text-xs text-slate-400 font-bold mt-1 uppercase tracking-widest">Analisi e Proiezioni Future</p>
                    </div>
                    
                    <div className="flex flex-wrap items-center gap-4">
                      <div className="flex bg-slate-100 p-1 rounded-xl border border-slate-200/50">
                        <button
                          onClick={() => setChartType('bar')}
                          className={cn(
                            "p-2 rounded-lg transition-all",
                            chartType === 'bar' ? "bg-white text-indigo-600 shadow-sm" : "text-slate-400 hover:text-slate-600"
                          )}
                        >
                          <BarChart3 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setChartType('line')}
                          className={cn(
                            "p-2 rounded-lg transition-all",
                            chartType === 'line' ? "bg-white text-indigo-600 shadow-sm" : "text-slate-400 hover:text-slate-600"
                          )}
                        >
                          <LineChartIcon className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setChartType('area')}
                          className={cn(
                            "p-2 rounded-lg transition-all",
                            chartType === 'area' ? "bg-white text-indigo-600 shadow-sm" : "text-slate-400 hover:text-slate-600"
                          )}
                        >
                          <Activity className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setChartType('pie')}
                          className={cn(
                            "p-2 rounded-lg transition-all",
                            chartType === 'pie' ? "bg-white text-indigo-600 shadow-sm" : "text-slate-400 hover:text-slate-600"
                          )}
                        >
                          <PieChartIcon className="w-4 h-4" />
                        </button>
                      </div>

                      {chartType !== 'pie' && (
                        <div className="flex items-center gap-3 bg-slate-50 px-4 py-2 rounded-xl border border-slate-200">
                          <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Proiezione: {projectionMonths} Mesi</label>
                          <input
                            type="range"
                            min="1"
                            max="24"
                            value={projectionMonths}
                            onChange={(e) => setProjectionMonths(parseInt(e.target.value))}
                            className="w-24 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                          />
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      {chartType === 'bar' ? (
                        <BarChart data={getProjectionData()} barGap={8}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 10, fontWeight: 700, fill: '#64748b' }} />
                          <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fontWeight: 500, fill: '#94a3b8' }} />
                          <Tooltip 
                            cursor={{ fill: '#f8fafc' }}
                            contentStyle={{ border: 'none', borderRadius: '20px', boxShadow: '0 10px 30px -5px rgba(0,0,0,0.05)', padding: '16px' }}
                          />
                          <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '10px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '1px' }} />
                          <Bar dataKey="income" name="Entrate" fill="#10b981" radius={[6, 6, 6, 6]} barSize={30} />
                          <Bar dataKey="expense" name="Uscite" fill="#f43f5e" radius={[6, 6, 6, 6]} barSize={30} />
                          <Bar dataKey="profit" name="Profitto" fill="#6366f1" radius={[6, 6, 6, 6]} barSize={30} />
                        </BarChart>
                      ) : chartType === 'line' ? (
                        <LineChart data={getProjectionData()}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 10, fontWeight: 700, fill: '#64748b' }} />
                          <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fontWeight: 500, fill: '#94a3b8' }} />
                          <Tooltip 
                            contentStyle={{ border: 'none', borderRadius: '20px', boxShadow: '0 10px 30px -5px rgba(0,0,0,0.05)', padding: '16px' }}
                          />
                          <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '10px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '1px' }} />
                          <Line type="monotone" dataKey="income" name="Entrate" stroke="#10b981" strokeWidth={4} dot={{ r: 4, strokeWidth: 2, fill: '#fff' }} activeDot={{ r: 6 }} />
                          <Line type="monotone" dataKey="expense" name="Uscite" stroke="#f43f5e" strokeWidth={4} dot={{ r: 4, strokeWidth: 2, fill: '#fff' }} activeDot={{ r: 6 }} />
                          <Line type="monotone" dataKey="profit" name="Profitto" stroke="#6366f1" strokeWidth={4} dot={{ r: 4, strokeWidth: 2, fill: '#fff' }} activeDot={{ r: 6 }} />
                        </LineChart>
                      ) : chartType === 'area' ? (
                        <AreaChart data={getProjectionData()}>
                          <defs>
                            <linearGradient id="colorIncome" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorExpense" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 10, fontWeight: 700, fill: '#64748b' }} />
                          <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fontWeight: 500, fill: '#94a3b8' }} />
                          <Tooltip 
                            contentStyle={{ border: 'none', borderRadius: '20px', boxShadow: '0 10px 30px -5px rgba(0,0,0,0.05)', padding: '16px' }}
                          />
                          <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '10px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '1px' }} />
                          <Area type="monotone" dataKey="income" name="Entrate" stroke="#10b981" fillOpacity={1} fill="url(#colorIncome)" strokeWidth={3} />
                          <Area type="monotone" dataKey="expense" name="Uscite" stroke="#f43f5e" fillOpacity={1} fill="url(#colorExpense)" strokeWidth={3} />
                          <Area type="monotone" dataKey="profit" name="Profitto" stroke="#6366f1" fillOpacity={1} fill="url(#colorProfit)" strokeWidth={3} />
                        </AreaChart>
                      ) : (
                        <PieChart>
                          <Pie
                            data={getPieData()}
                            cx="50%"
                            cy="50%"
                            innerRadius={80}
                            outerRadius={140}
                            paddingAngle={5}
                            dataKey="value"
                          >
                            {getPieData().map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={['#6366f1', '#f43f5e', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'][index % 6]} />
                            ))}
                          </Pie>
                          <Tooltip 
                            contentStyle={{ border: 'none', borderRadius: '20px', boxShadow: '0 10px 30px -5px rgba(0,0,0,0.05)', padding: '16px' }}
                          />
                          <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: '10px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '1px' }} />
                        </PieChart>
                      )}
                    </ResponsiveContainer>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'location' && (
              <motion.div
                key="location"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-8"
              >
                <div className="bg-white rounded-[40px] p-10 border border-slate-100 shadow-sm">
                  <div className="flex items-center gap-3 mb-8">
                    <Globe className="w-6 h-6 text-indigo-600" />
                    <h3 className="text-2xl font-extrabold text-slate-900 tracking-tight">Intelligence Territoriale</h3>
                  </div>
                  
                  <div className="space-y-6 mb-10">
                    <div className="flex flex-col md:flex-row gap-4">
                      <div className="relative flex-1">
                        <MapPin className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                        <input
                          type="text"
                          value={location}
                          onChange={(e) => setLocation(e.target.value)}
                          placeholder="Cerca una località o un indirizzo..."
                          className="w-full pl-14 pr-6 py-5 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 outline-none transition-all font-medium"
                        />
                      </div>
                      <div className="flex-1">
                        <select
                          value={businessType}
                          onChange={(e) => setBusinessType(e.target.value)}
                          className="w-full px-6 py-5 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 outline-none transition-all font-medium appearance-none"
                        >
                          <option value="all">Tutte le attività</option>
                          <option value="bar">Bar</option>
                          <option value="ristoranti">Ristoranti</option>
                          <option value="meccanici">Meccanici</option>
                          <option value="negozi">Negozi</option>
                          <option value="palestre">Palestre</option>
                        </select>
                      </div>
                    </div>

                    <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100">
                      <div className="flex items-center justify-between mb-4">
                        <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Raggio di Ricerca: {radius} KM</label>
                        <span className="text-xs font-bold text-indigo-600">Max 50 KM</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="50"
                        value={radius}
                        onChange={(e) => setRadius(parseInt(e.target.value))}
                        className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                      />
                    </div>

                    <button
                      onClick={handleLocationAnalyze}
                      disabled={isLocAnalyzing || !location.trim() || !idea.trim()}
                      className="w-full bg-indigo-600 text-white py-5 rounded-2xl font-bold hover:bg-indigo-700 transition-all disabled:opacity-50 flex items-center justify-center gap-3 shadow-lg shadow-indigo-100"
                    >
                      {isLocAnalyzing ? <Loader2 className="animate-spin w-5 h-5" /> : <BarChart3 className="w-5 h-5" />}
                      <span>Analizza Area e Cerca Sponsor</span>
                    </button>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Ricerca Sponsor Section */}
                    <div className="space-y-6">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-indigo-50 rounded-lg flex items-center justify-center">
                          <Users className="w-4 h-4 text-indigo-600" />
                        </div>
                        <h4 className="font-bold text-slate-900">Ricerca Sponsor</h4>
                      </div>
                      
                      <div className="min-h-[300px] p-6 rounded-3xl bg-slate-50 border border-slate-100">
                        {isLocAnalyzing ? (
                          <div className="h-full flex flex-col items-center justify-center py-20">
                            <Loader2 className="animate-spin w-8 h-8 text-indigo-400 mb-4" />
                            <p className="text-slate-400 text-sm font-medium">Scansione attività locali in corso...</p>
                          </div>
                        ) : locationAnalysis ? (
                          <div className="space-y-6">
                            <div className="prose prose-slate prose-sm max-w-none">
                              <Markdown>{locationAnalysis.text}</Markdown>
                            </div>
                            {locationAnalysis.sources.length > 0 && (
                              <div className="grid grid-cols-1 gap-3">
                                {locationAnalysis.sources.map((source: any, i: number) => (
                                  source.maps && (
                                    <a
                                      key={i}
                                      href={source.maps.uri}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="flex items-center gap-3 bg-white p-3 rounded-xl border border-slate-200 hover:border-indigo-500 transition-all"
                                    >
                                      <MapPin className="w-4 h-4 text-indigo-600" />
                                      <span className="text-xs font-bold text-slate-700">{source.maps.title}</span>
                                    </a>
                                  )
                                ))}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="h-full flex flex-col items-center justify-center py-20 text-center">
                            <Search className="w-10 h-10 text-slate-200 mb-4" />
                            <p className="text-slate-400 text-xs font-bold max-w-[200px]">Avvia l'analisi per trovare sponsor locali nel raggio scelto.</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Sponsor AI Section */}
                    <div className="space-y-6">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-purple-50 rounded-lg flex items-center justify-center">
                          <Sparkles className="w-4 h-4 text-purple-600" />
                        </div>
                        <h4 className="font-bold text-slate-900">Sponsor AI</h4>
                      </div>
                      
                      <div className="min-h-[300px] p-6 rounded-3xl bg-purple-50/30 border border-purple-100">
                        {/* Suggerimento iniziale basato sulla categoria */}
                        {!isSponsorAIAnalyzing && !sponsorAI && (
                          <div className="mb-6 p-4 bg-white/60 rounded-2xl border border-purple-100/50">
                            <div className="flex items-center gap-2 mb-2">
                              <Lightbulb className="w-3 h-3 text-purple-600" />
                              <span className="text-[10px] font-bold text-purple-600 uppercase tracking-widest">Consiglio Rapido</span>
                            </div>
                            <p className="text-xs text-slate-600 italic">
                              {projectCategory === 'evento' && "Per gli eventi, punta su Beverage (birrifici, energy drink), Automotive locali e Assicurazioni."}
                              {projectCategory === 'attività professionale' && "Per studi professionali, i partner ideali sono Software House, Banche e Fornitori di cancelleria/arredo ufficio."}
                              {projectCategory === 'corso' && "I corsi sono perfetti per sponsorizzazioni da parte di Tool di produttività, Librerie e Piattaforme Tech."}
                              {projectCategory === 'prodotto' && "Per i prodotti, cerca collaborazioni con Marketplace, Corrieri e Influencer di settore."}
                              {projectCategory === 'locale' && "Le attività locali funzionano bene con Fornitori alimentari, Servizi di Delivery e Radio locali."}
                            </p>
                          </div>
                        )}

                        {isSponsorAIAnalyzing ? (
                          <div className="h-full flex flex-col items-center justify-center py-20">
                            <div className="relative">
                              <div className="absolute inset-0 bg-purple-400 rounded-full blur-xl opacity-20 animate-pulse" />
                              <Sparkles className="w-8 h-8 text-purple-400 animate-spin-slow relative" />
                            </div>
                            <p className="text-purple-400 text-sm font-medium mt-4">L'AI sta profilando i migliori sponsor...</p>
                          </div>
                        ) : sponsorAI ? (
                          <div className="prose prose-purple prose-sm max-w-none">
                            <Markdown>{sponsorAI}</Markdown>
                          </div>
                        ) : (
                          <div className="h-full flex flex-col items-center justify-center py-20 text-center">
                            <Zap className="w-10 h-10 text-purple-200 mb-4" />
                            <p className="text-purple-400 text-xs font-bold max-w-[200px]">L'AI identificherà chi è più propenso a finanziare la tua idea.</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'chat' && (
              <motion.div
                key="chat"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="h-[700px] flex flex-col bg-white rounded-[40px] border border-slate-100 shadow-xl overflow-hidden"
              >
                <div className="p-6 border-b border-slate-50 flex items-center justify-between bg-slate-50/50">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "w-10 h-10 rounded-xl flex items-center justify-center shadow-md transition-colors",
                      chatPersona === 'marketing' ? "bg-indigo-600" : "bg-emerald-600"
                    )}>
                      {chatPersona === 'marketing' ? (
                        <MessageSquare className="text-white w-5 h-5" />
                      ) : (
                        <Scale className="text-white w-5 h-5" />
                      )}
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-slate-900">
                        {chatPersona === 'marketing' ? 'Strategy Buddy' : 'Commercialista AI'}
                      </h3>
                      <div className="flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                          {chatPersona === 'marketing' ? 'Esperto Marketing Online' : 'Esperto Legge Italiana Online'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex bg-slate-200 p-1 rounded-xl">
                    <button
                      onClick={() => setChatPersona('marketing')}
                      className={cn(
                        "px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all",
                        chatPersona === 'marketing' ? "bg-white text-indigo-600 shadow-sm" : "text-slate-500 hover:text-slate-700"
                      )}
                    >
                      Marketing
                    </button>
                    <button
                      onClick={() => setChatPersona('accountant')}
                      className={cn(
                        "px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all",
                        chatPersona === 'accountant' ? "bg-white text-emerald-600 shadow-sm" : "text-slate-500 hover:text-slate-700"
                      )}
                    >
                      Commercialista
                    </button>
                  </div>
                </div>

                <div className="flex-1 overflow-y-auto p-8 space-y-8 custom-scrollbar bg-slate-50/30">
                  {chatMessages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-center">
                      <div className="w-16 h-16 bg-white rounded-2xl shadow-sm flex items-center justify-center mb-6">
                        {chatPersona === 'marketing' ? (
                          <Sparkles className="w-8 h-8 text-indigo-400" />
                        ) : (
                          <Briefcase className="w-8 h-8 text-emerald-400" />
                        )}
                      </div>
                      <h4 className="text-lg font-bold text-slate-900 mb-2">
                        {chatPersona === 'marketing' ? 'Serve un secondo parere?' : 'Dubbi fiscali o legali?'}
                      </h4>
                      <p className="text-slate-500 text-sm max-w-xs mx-auto">
                        {chatPersona === 'marketing' 
                          ? "Chiedimi qualsiasi cosa sulla tua campagna, dal naming all'ottimizzazione del budget, nel rispetto della legge italiana."
                          : "Chiedimi informazioni su Partita IVA, regimi fiscali, detrazioni o normative per la tua attività in Italia."}
                      </p>
                    </div>
                  )}
                  {chatMessages.map((msg, i) => (
                    <div key={i} className={cn("flex", msg.role === 'user' ? "justify-end" : "justify-start")}>
                      <div className={cn(
                        "max-w-[85%] p-5 text-sm leading-relaxed shadow-sm",
                        msg.role === 'user' 
                          ? "bg-indigo-600 text-white rounded-3xl rounded-tr-none" 
                          : "bg-white border border-slate-100 text-slate-700 rounded-3xl rounded-tl-none prose prose-slate"
                      )}>
                        {msg.role === 'model' ? <Markdown>{msg.text}</Markdown> : msg.text}
                      </div>
                    </div>
                  ))}
                  {isChatLoading && (
                    <div className="flex justify-start">
                      <div className="bg-white p-5 rounded-3xl rounded-tl-none border border-slate-100">
                        <div className="flex gap-1">
                          <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" />
                          <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                          <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                <div className="p-6 border-t border-slate-50">
                  <form 
                    onSubmit={(e) => { e.preventDefault(); handleSendMessage(); }}
                    className="flex gap-3"
                  >
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder={chatPersona === 'marketing' ? "Chiedi al tuo marketing buddy..." : "Chiedi al tuo commercialista..."}
                      className="flex-1 p-5 bg-slate-50 rounded-2xl border border-slate-200 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/5 outline-none transition-all text-sm font-medium"
                    />
                    <button
                      type="submit"
                      disabled={!chatInput.trim() || isChatLoading}
                      className={cn(
                        "text-white px-8 rounded-2xl transition-all disabled:opacity-50 shadow-lg",
                        chatPersona === 'marketing' ? "bg-indigo-600 hover:bg-indigo-700 shadow-indigo-100" : "bg-emerald-600 hover:bg-emerald-700 shadow-emerald-100"
                      )}
                    >
                      <Send className="w-5 h-5" />
                    </button>
                  </form>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Modern Footer */}
      <footer className="max-w-7xl mx-auto px-10 py-20 mt-10 border-t border-slate-200/60 flex flex-col md:flex-row justify-between items-center gap-10">
        <div className="flex items-center gap-3 opacity-50">
          <Sparkles className="w-5 h-5 text-indigo-600" />
          <span className="text-xs font-bold uppercase tracking-widest text-slate-900">Marketing buddy AI</span>
        </div>
        <div className="flex gap-10 text-[10px] font-bold uppercase tracking-widest text-slate-400">
          <a href="#" className="hover:text-indigo-600 transition-colors">Intelligenza</a>
          <a href="#" className="hover:text-indigo-600 transition-colors">Strategia</a>
          <a href="#" className="hover:text-indigo-600 transition-colors">Crescita</a>
        </div>
      </footer>

      <AnimatePresence>
        {notification && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className={cn(
              "fixed bottom-10 left-1/2 -translate-x-1/2 z-[100] px-6 py-3 rounded-2xl shadow-2xl flex items-center gap-3 border",
              notification.type === 'success' ? "bg-emerald-600 border-emerald-500 text-white" : "bg-rose-600 border-rose-500 text-white"
            )}
          >
            {notification.type === 'success' ? <CheckCircle2 className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
            <span className="text-sm font-bold">{notification.message}</span>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showBudgetModal && (
          <div className="fixed inset-0 z-[110] flex items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowBudgetModal(false)}
              className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative w-full max-w-md bg-white rounded-[40px] p-10 shadow-2xl"
            >
              <h3 className="text-2xl font-black text-slate-900 mb-2">Limiti di Budget</h3>
              <p className="text-sm text-slate-500 mb-8">Imposta i tetti di spesa mensili per ogni categoria.</p>
              
              <div className="space-y-6">
                {categoryBudgets.map((budget) => (
                  <div key={budget.category} className="space-y-2">
                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest ml-1">{budget.category}</label>
                    <div className="relative">
                      <DollarSign className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                      <input
                        type="number"
                        value={budget.limit}
                        onChange={(e) => updateBudget(budget.category, parseFloat(e.target.value) || 0)}
                        className="w-full pl-12 pr-6 py-4 rounded-2xl bg-slate-50 border border-slate-200 focus:border-indigo-500 outline-none transition-all font-bold"
                      />
                    </div>
                  </div>
                ))}
              </div>

              <button
                onClick={() => setShowBudgetModal(false)}
                className="w-full mt-10 bg-indigo-600 text-white py-5 rounded-2xl font-bold hover:bg-indigo-700 transition-all shadow-lg shadow-indigo-100"
              >
                Salva Configurazioni
              </button>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #e2e8f0;
          border-radius: 20px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #cbd5e1;
        }
      `}</style>
    </div>
  );
}
