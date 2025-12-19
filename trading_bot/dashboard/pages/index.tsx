import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Activity, TrendingUp, TrendingDown, DollarSign, 
  Brain, Newspaper, BarChart3, Zap, AlertCircle,
  Eye, Clock, Wifi, WifiOff, X, Fish, ArrowUpRight, ArrowDownRight, XCircle,
  CheckCircle, Loader2
} from 'lucide-react'

interface WhaleTx {
  hash: string
  btc: number
  usd_approx: number
  from_exchange: string | null
  to_exchange: string | null
  time: string
  type: string
}

interface WhaleData {
  sentiment: string
  score: number
  deposits: number
  withdrawals: number
  depositBtc: number
  withdrawalBtc: number
  netFlow: number
  netFlowUsd: number
  reasoning: string
  largeTxs: WhaleTx[]
  hasAlert: boolean
  timestamp?: string
  date?: string
}

interface OverlordData {
  enabled: boolean
  decision: string
  confidence: number
  reasoning: string
  entryType: string
  riskLevel: string
  keyFactors: string[]
  cached: boolean
  model: string
}

interface BotData {
  connected: boolean
  mode: 'LIVE' | 'PAPER'
  symbol: string
  balance: number
  position: {
    side: 'LONG' | 'SHORT' | null
    quantity: number
    entryPrice: number
    pnl: number
    pnlPercent: number
  }
  signals: {
    ml: { direction: string, confidence: number, prediction: number }
    news: { sentiment: string, score: number, confidence: number }
    wisdom: { signal: string, grade: string, master: string, reasoning: string, specialty?: string, style?: string, selectionReason?: string, fullAnalysis?: any, keyLevels?: any }
    quant: { signal: string, zScore: number, momentum: number, htfTrend?: string, htfBias?: number, volatilityRegime?: string, volatilityPctl?: number, kellyFraction?: number, optimalSize?: number, reasoning?: string, momentum5p?: number, momentum10p?: number, momentum20p?: number }
  }
  whale: WhaleData
  whaleHistory: WhaleData[]
  costs: {
    commission: number
    funding: number
    total: number
  }
  screenshots: string[]
  lastUpdate: string
  currentPrice: number
  decision: { action: string, strength: number }
  entryMode?: {
    mode: string
    scaledEnabled: boolean
    aggressiveThreshold: number
    currentSignalStrength: number
  }
  learning: {
    totalTrades: number
    allTrades: number
    minForRetrain: number
    tradesUntilRetrain: number
    winRate: number
    lastRetrain: string | null
    readyToRetrain: boolean
    kellyFraction: number
    expectancy: number
    maxDrawdown: number
    sharpeRatio: number
    profitFactor: number
  }
  overlord?: OverlordData
  thinking?: {
    active: boolean
    stage: string
    stages: {
      ml: { status: string, result: any }
      news: { status: string, result: any }
      wisdom: { status: string, result: any }
      quant: { status: string, result: any }
      whale: { status: string, result: any }
      overlord: { status: string, result: any }
    }
    finalDecision: { action: string, strength: number } | null
  }
}

const defaultWhale: WhaleData = {
  sentiment: 'NEUTRAL',
  score: 0,
  deposits: 0,
  withdrawals: 0,
  depositBtc: 0,
  withdrawalBtc: 0,
  netFlow: 0,
  netFlowUsd: 0,
  reasoning: '',
  largeTxs: [],
  hasAlert: false
}

const defaultData: BotData = {
  connected: false,
  mode: 'LIVE',
  symbol: 'BTCUSDC',
  balance: 0,
  position: { side: null, quantity: 0, entryPrice: 0, pnl: 0, pnlPercent: 0 },
  signals: {
    ml: { direction: 'HOLD', confidence: 0, prediction: 0 },
    news: { sentiment: 'NEUTRAL', score: 0, confidence: 0 },
    wisdom: { signal: 'HOLD', grade: 'C', master: '', reasoning: '' },
    quant: { signal: 'HOLD', zScore: 0, momentum: 0, htfTrend: 'NEUTRAL', htfBias: 0 }
  },
  whale: defaultWhale,
  whaleHistory: [],
  costs: { commission: 0, funding: 0, total: 0 },
  screenshots: [],
  lastUpdate: '',
  currentPrice: 0,
  decision: { action: 'HOLD', strength: 0 },
  learning: { totalTrades: 0, allTrades: 0, minForRetrain: 50, tradesUntilRetrain: 50, winRate: 0, lastRetrain: null, readyToRetrain: false, kellyFraction: 0.5, expectancy: 0, maxDrawdown: 0, sharpeRatio: 0, profitFactor: 0 }
}

export default function Dashboard() {
  const [data, setData] = useState<BotData>(defaultData)
  const [logs, setLogs] = useState<string[]>([])
  const [showWhaleAlert, setShowWhaleAlert] = useState(false)
  const [showWhaleHistory, setShowWhaleHistory] = useState(false)
  const [showWisdomDetail, setShowWisdomDetail] = useState(false)
  const [showQuantDetail, setShowQuantDetail] = useState(false)
  const [whaleAlertDismissed, setWhaleAlertDismissed] = useState(false)
  const [isClosing, setIsClosing] = useState(false)
  const [particles, setParticles] = useState<Array<{left: string, top: string, delay: string, duration: string}>>([])
  const [robotText, setRobotText] = useState('')
  const [robotPhrase, setRobotPhrase] = useState(0)
  const lastWhaleNetFlow = useRef<number>(0)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const logsContainerRef = useRef<HTMLDivElement>(null)
  const userScrolledUp = useRef(false)
  

  useEffect(() => {
    setParticles([...Array(15)].map(() => ({
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      delay: `${Math.random() * 6}s`,
      duration: `${4 + Math.random() * 4}s`
    })))
  }, [])

  const robotPhrases = [
    'SEND IT',
    'FULL SIZE, NO STOP',
    'LOOKS WEAK, FADING THIS',
    'TRAPPED LONGS EVERYWHERE',
    'LIQUIDATIONS INCOMING',
    'THEY DONT KNOW',
    'EASY MONEY TODAY',
    'SIZE UP OR SHUT UP',
    'PRINTING HARD',
    'BEARS ARE COOKED',
    'BULLS IN SHAMBLES',
    'STOP HUNT COMPLETE',
    'SHAKING OUT WEAK HANDS',
    'REVERSAL LOADING',
    'THIS IS THE PLAY',
    'TRUST THE PROCESS',
    'FADE THE RETAIL',
    'SMART MONEY MOVING',
    'PATIENCE PAYS',
    'MAX PAIN INCOMING',
    "WHALES FISHING",
    ""
  ]
  
  useEffect(() => {
    let charIndex = 0
    let isDeleting = false
    let currentPhrase = robotPhrases[robotPhrase]
    
    const typeEffect = setInterval(() => {
      if (!isDeleting) {
        setRobotText(currentPhrase.slice(0, charIndex + 1))
        charIndex++
        if (charIndex === currentPhrase.length) {
          isDeleting = true
          setTimeout(() => {}, 2000) // Pause at end
        }
      } else {
        setRobotText(currentPhrase.slice(0, charIndex - 1))
        charIndex--
        if (charIndex === 0) {
          isDeleting = false
          setRobotPhrase((prev) => (prev + 1) % robotPhrases.length)
        }
      }
    }, isDeleting ? 50 : 100)
    
    return () => clearInterval(typeEffect)
  }, [robotPhrase])

  useEffect(() => {
    const currentNetFlow = data.whale?.netFlow || 0
    const hasNewAlert = data.whale?.hasAlert && 
                        Math.abs(currentNetFlow - lastWhaleNetFlow.current) > 10 &&
                        !whaleAlertDismissed
    
    if (hasNewAlert) {
      setShowWhaleAlert(true)
      lastWhaleNetFlow.current = currentNetFlow
      // Auto-hide after 30 seconds
      const timer = setTimeout(() => {
        setShowWhaleAlert(false)
        setWhaleAlertDismissed(true)
      }, 30000)
      return () => clearTimeout(timer)
    }
  }, [data.whale?.hasAlert, data.whale?.netFlow, whaleAlertDismissed])
  
  useEffect(() => {
    const currentNetFlow = data.whale?.netFlow || 0
    if (Math.abs(currentNetFlow - lastWhaleNetFlow.current) > 50) {
      setWhaleAlertDismissed(false)
    }
  }, [data.whale?.netFlow])

  const handleLogsScroll = () => {
    if (logsContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logsContainerRef.current
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 30
      userScrolledUp.current = !isAtBottom
    }
  }

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'

  const handleClosePosition = async () => {
    if (isClosing || !data.position.side) return
    
    if (!confirm(`Close ${data.position.side} position at market price?\nThis will also cancel TP/SL orders.`)) return
    
    setIsClosing(true)
    try {
      const res = await fetch(`${API_URL}/api/close-position`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      const result = await res.json()
      if (result.success) {
        alert(`✅ ${result.message}`)
      } else {
        alert(`❌ Error: ${result.error}`)
      }
    } catch (err) {
      alert(`❌ Failed to close: ${err}`)
    } finally {
      setIsClosing(false)
    }
  }

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/status')
        if (res.ok) {
          const json = await res.json()
          setData({ ...json, connected: true })
        }
      } catch (e) {
        setData(prev => ({ ...prev, connected: false }))
      }
    }

    const fetchLogs = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/logs')
        if (res.ok) {
          const json = await res.json()
          setLogs(json.logs || [])
        }
      } catch (e) {}
    }

    fetchData()
    fetchLogs()
    const interval = setInterval(() => {
      fetchData()
      fetchLogs()
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!userScrolledUp.current && logsEndRef.current && logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight
    }
  }, [logs])

  const getSignalColor = (signal: string) => {
    if (signal === 'LONG' || signal === 'BUY' || signal === 'BULLISH') return 'text-green-400'
    if (signal === 'SHORT' || signal === 'SELL' || signal === 'BEARISH') return 'text-red-400'
    return 'text-yellow-400'
  }

  const getGradeColor = (grade: string) => {
    if (grade === 'A') return 'text-green-400 bg-green-400/20'
    if (grade === 'B') return 'text-blue-400 bg-blue-400/20'
    if (grade === 'C') return 'text-yellow-400 bg-yellow-400/20'
    return 'text-red-400 bg-red-400/20'
  }

  return (
    <div className="min-h-screen animated-bg text-white p-4 relative overflow-hidden">
      {/* Floating particles */}
      <div className="absolute inset-0 pointer-events-none">
        {particles.map((p, i) => (
          <div
            key={i}
            className="particle"
            style={{
              left: p.left,
              top: p.top,
              animationDelay: p.delay,
              animationDuration: p.duration
            }}
          />
        ))}
      </div>

      {/* Header */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-xl p-4 mb-4 border-glow relative z-10"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <motion.div 
              className="w-[68px] h-[68px] rounded-full overflow-hidden border-2 border-[#00d4aa] shadow-lg shadow-[#00d4aa]/30"
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              <img src="/RocketRatLogo.jpg" alt="RocketRat" className="w-full h-full object-cover" />
            </motion.div>
            <div>
              <h1 className="text-2xl font-bold gradient-text">RocketRat</h1>
              <p className="text-gray-400 text-sm">The Ultimate Prop Trader</p>
            </div>
          </div>
          
          {/* Animated Robot Text */}
          <div className="flex-1 flex justify-center">
            <div className="px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500/10 via-purple-500/10 to-pink-500/10 border border-cyan-500/20">
              <motion.div 
                className="font-mono text-sm tracking-wider"
                animate={{ 
                  textShadow: ['0 0 10px #00d4aa', '0 0 20px #00d4aa', '0 0 10px #00d4aa']
                }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <span className="text-cyan-400">&gt;_</span>
                <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent font-bold">
                  {robotText}
                </span>
                <motion.span 
                  className="text-cyan-400 ml-0.5"
                  animate={{ opacity: [1, 0, 1] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                >
                  |
                </motion.span>
              </motion.div>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              {data.connected ? (
                <div className="relative">
                  <Wifi className="w-5 h-5 text-green-400" />
                  <span className="absolute -top-1 -right-1 w-2 h-2 bg-green-400 rounded-full live-pulse" />
                </div>
              ) : (
                <WifiOff className="w-5 h-5 text-red-400" />
              )}
              <span className={data.connected ? 'text-green-400' : 'text-red-400'}>
                {data.connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <motion.div 
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                data.mode === 'LIVE' ? 'bg-red-500/20 text-red-400 neon-red' : 'bg-blue-500/20 text-blue-400'
              }`}
              animate={data.mode === 'LIVE' ? { scale: [1, 1.05, 1] } : {}}
              transition={{ duration: 2, repeat: Infinity }}
            >
              {data.mode === 'LIVE' ? '🔴 LIVE' : '🔵 PAPER'}
            </motion.div>
            {/* Entry Mode Badge */}
            {data.entryMode && (
              <div 
                className={`px-3 py-1 rounded-full text-sm font-medium cursor-help ${
                  data.entryMode.mode === 'auto' 
                    ? (data.entryMode.currentSignalStrength >= data.entryMode.aggressiveThreshold 
                        ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' 
                        : 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30')
                    : data.entryMode.mode === 'aggressive'
                      ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                      : 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                }`}
                title={data.entryMode.mode === 'auto' 
                  ? (data.entryMode.currentSignalStrength >= data.entryMode.aggressiveThreshold 
                      ? 'Fast Entry: Strong signal detected - will enter quickly after 50% filled' 
                      : 'Patient Entry: Weak signal - will wait for all limit orders to fill for better price')
                  : data.entryMode.mode === 'aggressive' 
                    ? 'Fast Entry: Will enter quickly after 50% of orders filled'
                    : 'Patient Entry: Will wait for all limit orders to fill'}
              >
                {data.entryMode.mode === 'auto' ? (
                  data.entryMode.currentSignalStrength >= data.entryMode.aggressiveThreshold 
                    ? '⚡ Fast Entry' 
                    : '🎯 Patient Entry'
                ) : data.entryMode.mode === 'aggressive' ? '⚡ Fast Entry' : '🎯 Patient Entry'}
              </div>
            )}
            <div className="text-right">
              <p className="text-gray-400 text-xs">Balance</p>
              <motion.p 
                className="text-xl font-bold"
                key={data.balance}
                initial={{ scale: 1.1 }}
                animate={{ scale: 1 }}
              >
                ${data.balance.toFixed(2)}
              </motion.p>
            </div>
            {/* ML Learning Status */}
            <div className="ml-4 px-3 py-2 rounded-lg bg-purple-500/10 border border-purple-500/30">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-400" />
                <span className="text-xs text-gray-400">ML Retrain</span>
                {(data.learning?.allTrades || 0) > 0 && (
                  <span className="text-xs text-gray-500">({data.learning?.totalTrades || 0}/{data.learning?.allTrades || 0} closed)</span>
                )}
              </div>
              <div className="flex items-center gap-2 mt-1">
                {data.learning?.readyToRetrain ? (
                  <span className="text-green-400 text-sm font-bold">✓ Ready</span>
                ) : (
                  <>
                    <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                        style={{ width: `${Math.min(100, ((data.learning?.totalTrades || 0) / (data.learning?.minForRetrain || 50)) * 100)}%` }}
                      />
                    </div>
                    <span className="text-xs text-purple-300">{data.learning?.tradesUntilRetrain || 50} left</span>
                  </>
                )}
              </div>
            </div>
            {/* Risk Metrics */}
            {(data.learning?.totalTrades || 0) >= 5 && (
              <div className="ml-2 px-3 py-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
                <div className="flex items-center gap-3">
                  <div className="text-center">
                    <span className="text-xs text-gray-400 block">Kelly</span>
                    <span className="text-sm font-bold text-cyan-400">{((data.learning?.kellyFraction || 0.5) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="text-center">
                    <span className="text-xs text-gray-400 block">EV</span>
                    <span className={`text-sm font-bold ${(data.learning?.expectancy || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {(data.learning?.expectancy || 0).toFixed(2)}%
                    </span>
                  </div>
                  <div className="text-center">
                    <span className="text-xs text-gray-400 block">MDD</span>
                    <span className="text-sm font-bold text-orange-400">{(data.learning?.maxDrawdown || 0).toFixed(1)}%</span>
                  </div>
                  <div className="text-center">
                    <span className="text-xs text-gray-400 block">Sharpe</span>
                    <span className={`text-sm font-bold ${(data.learning?.sharpeRatio || 0) >= 1 ? 'text-green-400' : 'text-yellow-400'}`}>
                      {(data.learning?.sharpeRatio || 0).toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </motion.header>

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-8 space-y-4">
          {/* TradingView Chart */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass rounded-xl overflow-hidden"
            style={{ height: '400px' }}
          >
            <iframe
              src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_widget&symbol=BINANCE%3ABTCUSDC&interval=15&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=%5B%5D&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1&showpopupbutton=1&studies_overrides=%7B%7D&overrides=%7B%7D&enabled_features=%5B%5D&disabled_features=%5B%5D&showsymbolsearch=1"
              style={{ width: '100%', height: '100%', border: 'none' }}
              allowFullScreen
            />
          </motion.div>

          {/* LLM Overlord - Under Chart */}
          <motion.div 
            className="overlord-card overlord-glow rounded-xl p-5"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.05 }}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="overlord-text text-lg font-bold flex items-center gap-2">
                <span className="overlord-icon text-2xl">🤖</span>
                OVERLORD
              </h3>
              {data.overlord?.cached && (
                <span className="text-xs px-2 py-1 bg-gray-700/50 rounded text-gray-400">cached</span>
              )}
            </div>
            
            {data.overlord ? (
              <div className="space-y-3">
                {/* Decision & Confidence */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                      data.overlord.decision === 'LONG' ? 'bg-green-500/20 ring-2 ring-green-500' : 
                      data.overlord.decision === 'SHORT' ? 'bg-red-500/20 ring-2 ring-red-500' : 
                      'bg-yellow-500/20 ring-2 ring-yellow-500'
                    }`}>
                      {data.overlord.decision === 'LONG' ? (
                        <TrendingUp className="w-6 h-6 text-green-400" />
                      ) : data.overlord.decision === 'SHORT' ? (
                        <TrendingDown className="w-6 h-6 text-red-400" />
                      ) : (
                        <Activity className="w-6 h-6 text-yellow-400" />
                      )}
                    </div>
                    <div>
                      <span className={`text-2xl font-bold ${
                        data.overlord.decision === 'LONG' ? 'text-green-400' : 
                        data.overlord.decision === 'SHORT' ? 'text-red-400' : 'text-yellow-400'
                      }`}>
                        {data.overlord.decision}
                      </span>
                      <p className="text-xs text-gray-500">{data.overlord.entryType} Entry</p>
                    </div>
                  </div>
                  
                  {/* Confidence Ring */}
                  <div className="confidence-ring flex flex-col items-center">
                    <svg width="60" height="60" viewBox="0 0 60 60">
                      <circle cx="30" cy="30" r="26" fill="none" stroke="#374151" strokeWidth="4" />
                      <circle 
                        cx="30" cy="30" r="26" fill="none" 
                        stroke={data.overlord.confidence >= 70 ? '#22c55e' : data.overlord.confidence >= 50 ? '#eab308' : '#ef4444'}
                        strokeWidth="4"
                        strokeDasharray={`${(data.overlord.confidence / 100) * 163.36} 163.36`}
                        strokeLinecap="round"
                      />
                    </svg>
                    <span className="absolute text-sm font-bold text-white" style={{marginTop: '18px'}}>
                      {data.overlord.confidence}%
                    </span>
                  </div>
                </div>

                {/* Risk Badge */}
                <div className="flex items-center gap-2">
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    data.overlord.riskLevel === 'LOW' ? 'bg-green-500/20 text-green-400 ring-1 ring-green-500/50' :
                    data.overlord.riskLevel === 'HIGH' ? 'bg-red-500/20 text-red-400 ring-1 ring-red-500/50' :
                    'bg-yellow-500/20 text-yellow-400 ring-1 ring-yellow-500/50'
                  }`}>
                    {data.overlord.riskLevel} RISK
                  </span>
                </div>

                {/* Reasoning */}
                {data.overlord.reasoning && (
                  <p className="text-sm text-gray-400 leading-relaxed line-clamp-2">
                    {data.overlord.reasoning}
                  </p>
                )}

                {/* Key Factors */}
                {data.overlord.keyFactors && data.overlord.keyFactors.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {data.overlord.keyFactors.map((factor, i) => (
                      <span key={i} className="text-xs px-2 py-1 bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
                        {factor}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-6">
                <div className="animate-pulse">
                  <Brain className="w-10 h-10 text-purple-400/50 mx-auto mb-2" />
                  <p className="text-gray-500 text-sm">Awaiting decision...</p>
                </div>
              </div>
            )}
          </motion.div>
        </div>

        {/* Position & Decision Panel */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="col-span-4 space-y-4"
        >
          {/* Current Position */}
          <motion.div 
            className={`glass rounded-xl p-4 card-hover relative overflow-hidden ${
              data.position.side === 'LONG' ? 'neon-green' : 
              data.position.side === 'SHORT' ? 'neon-red' : ''
            }`}
            animate={data.position.side ? { boxShadow: ['0 0 20px rgba(0,255,136,0.3)', '0 0 40px rgba(0,255,136,0.5)', '0 0 20px rgba(0,255,136,0.3)'] } : {}}
            transition={{ duration: 2, repeat: Infinity }}
          >
            {data.position.side && <div className="scanner" />}
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-gray-400 text-sm font-medium">Current Position</h3>
              {data.position.side && (
                <motion.span 
                  className={`px-2 py-0.5 rounded text-xs font-bold ${
                    data.position.side === 'LONG' ? 'signal-long' : 'signal-short'
                  }`}
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  {data.position.side}
                </motion.span>
              )}
            </div>
            {data.position.side ? (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Size</span>
                  <span className="font-mono">{data.position.quantity.toFixed(4)} BTC</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Entry</span>
                  <span className="font-mono">${data.position.entryPrice.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">P&L</span>
                  <motion.span 
                    className={`font-mono font-bold text-lg ${
                      data.position.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'
                    }`}
                    key={data.position.pnlPercent}
                    initial={{ scale: 1.2 }}
                    animate={{ scale: 1 }}
                  >
                    {data.position.pnl >= 0 ? '+' : ''}{data.position.pnlPercent.toFixed(2)}%
                  </motion.span>
                </div>
                {/* Progress bar showing PnL */}
                <div className="mt-2 h-1 bg-gray-700 rounded-full overflow-hidden">
                  <motion.div 
                    className={`h-full ${data.position.pnl >= 0 ? 'bg-green-400' : 'bg-red-400'}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(Math.abs(data.position.pnlPercent) * 10, 100)}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
                {/* Emergency Close Button - Missile Launch Style */}
                <div className="mt-4 flex justify-center">
                  <motion.button
                    onClick={handleClosePosition}
                    disabled={isClosing}
                    className="relative group"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {/* Outer ring - pulsing */}
                    <motion.div 
                      className="absolute inset-0 rounded-full bg-red-600/30 blur-md"
                      animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.8, 0.5] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    />
                    {/* Button housing */}
                    <div className="relative w-20 h-20 rounded-full bg-gradient-to-b from-gray-700 to-gray-900 
                                    p-1.5 shadow-[0_0_20px_rgba(239,68,68,0.5),inset_0_2px_10px_rgba(0,0,0,0.5)]
                                    border-4 border-gray-600">
                      {/* Inner button */}
                      <div className={`w-full h-full rounded-full flex items-center justify-center
                                      ${isClosing 
                                        ? 'bg-gradient-to-b from-gray-500 to-gray-700' 
                                        : 'bg-gradient-to-b from-red-500 to-red-700 hover:from-red-400 hover:to-red-600'}
                                      shadow-[inset_0_-4px_10px_rgba(0,0,0,0.4),inset_0_4px_10px_rgba(255,255,255,0.1)]
                                      transition-all duration-200 cursor-pointer
                                      active:shadow-[inset_0_4px_10px_rgba(0,0,0,0.6)]`}>
                        {isClosing ? (
                          <div className="w-6 h-6 border-3 border-white border-t-transparent rounded-full animate-spin" />
                        ) : (
                          <XCircle className="w-8 h-8 text-white drop-shadow-lg" />
                        )}
                      </div>
                    </div>
                    {/* Label */}
                    <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 whitespace-nowrap">
                      <span className="text-xs font-bold text-red-400 uppercase tracking-wider">
                        {isClosing ? 'CLOSING...' : 'EMERGENCY CLOSE'}
                      </span>
                    </div>
                  </motion.button>
                </div>
                <div className="h-6" /> {/* Spacer for label */}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-4">No Position</p>
            )}
          </motion.div>

          {/* Final Decision */}
          <div className="glass rounded-xl p-4">
            <h3 className="text-gray-400 text-sm font-medium mb-3">Final Decision</h3>
            <div className="flex items-center justify-center gap-3">
              {data.decision.action === 'LONG' ? (
                <TrendingUp className="w-8 h-8 text-green-400" />
              ) : data.decision.action === 'SHORT' ? (
                <TrendingDown className="w-8 h-8 text-red-400" />
              ) : (
                <Activity className="w-8 h-8 text-yellow-400" />
              )}
              <span className={`text-3xl font-bold ${getSignalColor(data.decision.action)}`}>
                {data.decision.action}
              </span>
            </div>
            <div className="mt-2 text-center text-gray-400 text-sm">
              Strength: {(data.decision.strength * 100).toFixed(1)}%
            </div>
          </div>

          {/* Trading Costs */}
          <div className="glass rounded-xl p-4">
            <h3 className="text-gray-400 text-sm font-medium mb-3 flex items-center gap-2">
              <DollarSign className="w-4 h-4" /> Costs (24h)
            </h3>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <p className="text-xs text-gray-500">Commission</p>
                <p className="font-mono text-sm">${data.costs.commission.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Funding</p>
                <p className={`font-mono text-sm ${data.costs.funding >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {data.costs.funding >= 0 ? '+' : ''}{data.costs.funding.toFixed(4)}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Total</p>
                <p className="font-mono text-sm text-red-400">${data.costs.total.toFixed(4)}</p>
              </div>
            </div>
          </div>

          {/* Trading Music */}
          <div className="glass rounded-xl p-3">
            <h3 className="text-gray-400 text-xs font-medium mb-2 flex items-center gap-2">
              🎵 Trading Vibes
            </h3>
            <div className="rounded-lg overflow-hidden">
              <iframe 
                width="100%" 
                height="80" 
                src="https://www.youtube.com/embed/MWkIxYtB8Ag?autoplay=0" 
                title="Trading Music"
                frameBorder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowFullScreen
                className="rounded-lg"
              />
            </div>
          </div>
        </motion.div>

        {/* Signal Cards */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="col-span-12 grid grid-cols-4 gap-4 relative z-10"
        >
          {/* ML Signal */}
          <motion.div 
            className="glass rounded-xl p-4 card-hover relative overflow-hidden"
            whileHover={{ scale: 1.02 }}
          >
            <div className="absolute inset-0 opacity-10 bg-gradient-to-br from-purple-500 to-transparent" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-3">
                <motion.div animate={{ rotate: 360 }} transition={{ duration: 10, repeat: Infinity, ease: "linear" }}>
                  <BarChart3 className="w-5 h-5 text-purple-400" />
                </motion.div>
                <h3 className="font-medium">ML Prediction</h3>
              </div>
              <motion.div 
                className={`text-2xl font-bold ${getSignalColor(data.signals.ml.direction)}`}
                key={data.signals.ml.direction}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
              >
                {data.signals.ml.direction}
              </motion.div>
              <div className="mt-2 text-sm text-gray-400">
                Confidence: {data.signals.ml.confidence}%
              </div>
              {/* Confidence bar */}
              <div className="mt-2 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <motion.div 
                  className="h-full bg-purple-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${data.signals.ml.confidence}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </div>
              <div className="text-xs text-gray-500 mt-2">
                Predicted: ${data.signals.ml.prediction.toFixed(2)}
              </div>
            </div>
          </motion.div>

          {/* News Signal */}
          <motion.div 
            className="glass rounded-xl p-4 card-hover relative overflow-hidden"
            whileHover={{ scale: 1.02 }}
          >
            <div className="absolute inset-0 opacity-10 bg-gradient-to-br from-blue-500 to-transparent" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-3">
                <Newspaper className="w-5 h-5 text-blue-400" />
                <h3 className="font-medium">News Sentiment</h3>
              </div>
              <motion.div 
                className={`text-2xl font-bold ${getSignalColor(data.signals.news.sentiment)}`}
                key={data.signals.news.sentiment}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
              >
                {data.signals.news.sentiment}
              </motion.div>
              <div className="mt-2 text-sm text-gray-400">
                Score: {data.signals.news.score >= 0 ? '+' : ''}{data.signals.news.score.toFixed(2)}
              </div>
              {/* Sentiment bar */}
              <div className="mt-2 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <motion.div 
                  className={`h-full ${data.signals.news.score >= 0 ? 'bg-green-500' : 'bg-red-500'}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(Math.abs(data.signals.news.score) * 100, 100)}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </div>
              <div className="text-xs text-gray-500 mt-2">
                Confidence: {data.signals.news.confidence}%
              </div>
            </div>
          </motion.div>

          {/* Wisdom Signal - Compact Display (Click to expand) */}
          <motion.div 
            className="glass rounded-xl p-4 card-hover relative overflow-hidden cursor-pointer"
            whileHover={{ scale: 1.02 }}
            onClick={() => setShowWisdomDetail(true)}
          >
            <div className="absolute inset-0 opacity-10 bg-gradient-to-br from-[#00d4aa] via-purple-500 to-transparent" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-3">
                <Brain className="w-5 h-5 text-[#00d4aa]" />
                <h3 className="font-medium">Wisdom Oracle</h3>
                <motion.span 
                  className={`ml-auto px-2 py-0.5 rounded text-xs font-bold ${
                    data.signals.wisdom.grade.includes('A') ? 'grade-a' : 
                    data.signals.wisdom.grade === 'B' ? 'grade-b' : 'grade-c'
                  }`}
                  animate={data.signals.wisdom.grade.includes('A') ? { scale: [1, 1.1, 1] } : {}}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  {data.signals.wisdom.grade}
                </motion.span>
              </div>
              
              {/* Compact Trader Display */}
              <div className="flex items-center gap-3">
                <motion.div 
                  className="w-10 h-10 rounded-full bg-gradient-to-br from-[#00d4aa] to-purple-600 flex items-center justify-center text-xl flex-shrink-0"
                  animate={{ rotate: [0, 5, -5, 0] }}
                  transition={{ duration: 4, repeat: Infinity }}
                >
                  {data.signals.wisdom.master?.includes('Livermore') ? '📈' :
                   data.signals.wisdom.master?.includes('Soros') ? '🌍' :
                   data.signals.wisdom.master?.includes('Jones') ? '📊' :
                   data.signals.wisdom.master?.includes('Buffett') ? '🦉' :
                   data.signals.wisdom.master?.includes('Dalio') ? '⚙️' : '🧙'}
                </motion.div>
                <div className="flex-1 min-w-0">
                  <motion.div 
                    className="text-sm font-bold text-white truncate"
                    key={data.signals.wisdom.master}
                  >
                    {data.signals.wisdom.master || 'Analyzing...'}
                  </motion.div>
                  <div className="text-xs text-[#00d4aa] truncate">
                    {data.signals.wisdom.specialty || 'Market Analysis'}
                  </div>
                </div>
              </div>
              
              <motion.div 
                className={`text-2xl font-bold mt-2 ${getSignalColor(data.signals.wisdom.signal)}`}
                key={data.signals.wisdom.signal}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
              >
                {data.signals.wisdom.signal}
              </motion.div>
              <div className={`text-xs px-2 py-0.5 rounded inline-block mt-1 ${
                data.signals.wisdom.style === 'aggressive' ? 'bg-red-500/20 text-red-400' :
                data.signals.wisdom.style === 'contrarian' ? 'bg-purple-500/20 text-purple-400' :
                data.signals.wisdom.style === 'technical' ? 'bg-blue-500/20 text-blue-400' :
                data.signals.wisdom.style === 'patient' ? 'bg-green-500/20 text-green-400' :
                'bg-gray-500/20 text-gray-400'
              }`}>
                {data.signals.wisdom.style?.toUpperCase() || 'ANALYZING'}
              </div>
              <div className="text-xs text-gray-500 mt-2">Click for details →</div>
            </div>
          </motion.div>

          {/* Quant Signal */}
          <motion.div 
            className="glass rounded-xl p-4 card-hover relative overflow-hidden cursor-pointer"
            whileHover={{ scale: 1.02 }}
            onClick={() => setShowQuantDetail(true)}
          >
            <div className="absolute inset-0 opacity-10 bg-gradient-to-br from-yellow-500 to-transparent" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-3">
                <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ duration: 2, repeat: Infinity }}>
                  <Zap className="w-5 h-5 text-yellow-400" />
                </motion.div>
                <h3 className="font-medium">Quant Analysis</h3>
              </div>
              <motion.div 
                className={`text-2xl font-bold ${getSignalColor(data.signals.quant.signal)}`}
                key={data.signals.quant.signal}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
              >
                {data.signals.quant.signal}
              </motion.div>
              <div className="mt-2 text-sm text-gray-400">
                Z-Score: {data.signals.quant.zScore >= 0 ? '+' : ''}{data.signals.quant.zScore.toFixed(2)}
              </div>
              {/* Z-Score visualization */}
              <div className="mt-2 h-1.5 bg-gray-700 rounded-full overflow-hidden relative">
                <div className="absolute left-1/2 w-0.5 h-full bg-gray-500" />
                <motion.div 
                  className={`h-full ${data.signals.quant.zScore >= 0 ? 'bg-green-500 ml-[50%]' : 'bg-red-500'}`}
                  style={{ 
                    width: `${Math.min(Math.abs(data.signals.quant.zScore) * 16, 50)}%`,
                    marginLeft: data.signals.quant.zScore < 0 ? `${50 - Math.min(Math.abs(data.signals.quant.zScore) * 16, 50)}%` : '50%'
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(Math.abs(data.signals.quant.zScore) * 16, 50)}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </div>
              <div className="text-xs text-gray-500 mt-2">
                Momentum: {data.signals.quant.momentum >= 0 ? '+' : ''}{data.signals.quant.momentum.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">Click for details →</div>
            </div>
          </motion.div>
        </motion.div>

        {/* Screenshots & Wisdom Reasoning */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="col-span-5 glass rounded-xl p-4"
        >
          <div className="flex items-center gap-2 mb-3">
            <Eye className="w-5 h-5 text-[#00d4aa]" />
            <h3 className="font-medium">News Screenshots</h3>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {data.screenshots.length > 0 ? (
              data.screenshots.map((url, i) => (
                <motion.div 
                  key={i}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="aspect-video rounded-lg overflow-hidden bg-gray-800 cursor-pointer hover:ring-2 ring-[#00d4aa] transition-all"
                  onClick={() => window.open(`http://localhost:5000${url}`, '_blank')}
                >
                  <img 
                    src={`http://localhost:5000${url}`} 
                    alt={`Screenshot ${i + 1}`}
                    className="w-full h-full object-cover"
                  />
                </motion.div>
              ))
            ) : (
              <div className="col-span-3 text-center py-8 text-gray-500">
                No screenshots available
              </div>
            )}
          </div>
        </motion.div>

        {/* Wisdom Reasoning */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="col-span-7 glass rounded-xl p-4"
        >
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-5 h-5 text-[#00d4aa]" />
            <h3 className="font-medium">Wisdom Oracle Analysis</h3>
          </div>
          <div className="bg-[#12121a] rounded-lg p-4 text-sm text-gray-300 leading-relaxed max-h-32 overflow-y-auto">
            {data.signals.wisdom.reasoning || 'Waiting for analysis...'}
          </div>
        </motion.div>

        {/* Live Logs */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="col-span-12 glass rounded-xl p-4"
        >
          <div className="flex items-center gap-2 mb-3">
            <Clock className="w-5 h-5 text-[#00d4aa]" />
            <h3 className="font-medium">Live Output</h3>
            <div className="ml-auto flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400 live-pulse" />
              <span className="text-xs text-gray-400">{data.lastUpdate || 'Waiting...'}</span>
            </div>
          </div>
          <div 
            ref={logsContainerRef}
            onScroll={handleLogsScroll}
            className="bg-[#050508] rounded-lg p-3 font-mono text-xs max-h-48 overflow-y-auto"
          >
            {logs.length > 0 ? (
              logs.slice(-50).map((log, i) => (
                <div key={i} className={`py-0.5 ${
                  log.includes('LONG') || log.includes('BUY') || log.includes('BULLISH') ? 'text-green-400' :
                  log.includes('SHORT') || log.includes('SELL') || log.includes('BEARISH') ? 'text-red-400' :
                  log.includes('>>') ? 'text-[#00d4aa]' :
                  log.includes('Error') || log.includes('Failed') ? 'text-red-500' :
                  'text-gray-400'
                }`}>
                  {log}
                </div>
              ))
            ) : (
              <div className="text-gray-500 text-center py-4">Waiting for bot output...</div>
            )}
            <div ref={logsEndRef} />
          </div>
        </motion.div>
      </div>

      {/* Whale Alert Button - Always visible */}
      <motion.button
        onClick={() => setShowWhaleHistory(true)}
        className={`fixed bottom-4 right-4 z-40 flex items-center gap-2 px-4 py-3 rounded-xl ${
          data.whale?.hasAlert 
            ? 'bg-red-600/90 whale-alert-pulse' 
            : data.whale?.sentiment === 'BULLISH' 
              ? 'bg-green-600/80' 
              : data.whale?.sentiment === 'BEARISH'
                ? 'bg-red-600/80'
                : 'bg-gray-700/80'
        } backdrop-blur-sm shadow-lg hover:scale-105 transition-transform`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <Fish className="w-5 h-5" />
        <span className="font-medium">🐋 Whale Activity</span>
        {data.whale?.hasAlert && (
          <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-ping" />
        )}
      </motion.button>

      {/* Thinking Overlay - Live Decision Process */}
      <AnimatePresence>
        {data.thinking?.active && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative max-w-xl w-full mx-4 p-6 rounded-2xl bg-gradient-to-br from-purple-900/95 to-indigo-900/95 border-2 border-purple-500/50 shadow-2xl"
            >
              {/* Header */}
              <div className="text-center mb-6">
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  className="inline-block mb-3 w-16 h-16 rounded-full overflow-hidden border-2 border-purple-400 shadow-lg shadow-purple-500/50"
                >
                  <img src="/RocketRatLogo.jpg" alt="RocketRat" className="w-full h-full object-cover" />
                </motion.div>
                <h2 className="text-2xl font-bold text-white">🧠 Bot Thinking...</h2>
                <p className="text-purple-300 text-sm mt-1">Analyzing market conditions</p>
              </div>

              {/* Stages */}
              <div className="space-y-3">
                {[
                  { key: 'ml', icon: '📊', label: 'ML Prediction', color: 'blue' },
                  { key: 'news', icon: '📰', label: 'News Sentiment', color: 'yellow' },
                  { key: 'wisdom', icon: '🧙', label: 'Trading Wisdom', color: 'purple' },
                  { key: 'quant', icon: '📐', label: 'Quant Analysis', color: 'cyan' },
                  { key: 'whale', icon: '🐋', label: 'Whale Activity', color: 'green' },
                  { key: 'overlord', icon: '🤖', label: 'LLM Overlord', color: 'red' },
                ].map(({ key, icon, label, color }) => {
                  const stage = data.thinking?.stages?.[key as keyof typeof data.thinking.stages]
                  const status = stage?.status || 'pending'
                  const result = stage?.result
                  
                  return (
                    <motion.div
                      key={key}
                      initial={{ x: -20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      className={`flex items-center gap-3 p-3 rounded-lg ${
                        status === 'running' ? `bg-${color}-500/20 border border-${color}-500/50` :
                        status === 'complete' ? 'bg-green-500/20 border border-green-500/30' :
                        'bg-gray-800/50 border border-gray-700/50'
                      }`}
                    >
                      <span className="text-xl">{icon}</span>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-white">{label}</span>
                          {status === 'running' && (
                            <motion.div
                              animate={{ opacity: [0.5, 1, 0.5] }}
                              transition={{ duration: 1, repeat: Infinity }}
                              className="text-yellow-400 text-sm"
                            >
                              Processing...
                            </motion.div>
                          )}
                          {status === 'complete' && result && (
                            <span className={`text-sm font-bold ${
                              result.direction === 'LONG' || result.sentiment === 'BULLISH' ? 'text-green-400' :
                              result.direction === 'SHORT' || result.sentiment === 'BEARISH' ? 'text-red-400' :
                              result.decision === 'LONG' ? 'text-green-400' :
                              result.decision === 'SHORT' ? 'text-red-400' :
                              'text-gray-400'
                            }`}>
                              {result.direction || result.sentiment || result.decision || 'HOLD'}
                              {result.confidence && ` (${Math.round(result.confidence * 100)}%)`}
                              {result.master && ` - ${result.master}`}
                            </span>
                          )}
                          {status === 'pending' && (
                            <span className="text-gray-500 text-sm">Waiting...</span>
                          )}
                        </div>
                        {status === 'running' && result?.message && (
                          <p className="text-xs text-gray-400 mt-1">{result.message}</p>
                        )}
                      </div>
                      {status === 'complete' && <CheckCircle className="w-5 h-5 text-green-400" />}
                      {status === 'running' && (
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        >
                          <Loader2 className="w-5 h-5 text-yellow-400" />
                        </motion.div>
                      )}
                    </motion.div>
                  )
                })}
              </div>

              {/* Final Decision */}
              {data.thinking?.finalDecision && (
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="mt-6 p-4 rounded-xl bg-gradient-to-r from-purple-600/30 to-pink-600/30 border border-purple-500/50"
                >
                  <div className="text-center">
                    <p className="text-gray-300 text-sm mb-1">Final Decision</p>
                    <p className={`text-3xl font-bold ${
                      data.thinking.finalDecision.action === 'LONG' ? 'text-green-400' :
                      data.thinking.finalDecision.action === 'SHORT' ? 'text-red-400' :
                      'text-gray-400'
                    }`}>
                      {data.thinking.finalDecision.action === 'LONG' ? '🟢 LONG' :
                       data.thinking.finalDecision.action === 'SHORT' ? '🔴 SHORT' :
                       '⚪ HOLD'}
                    </p>
                    <p className="text-purple-300 text-sm mt-1">
                      Strength: {(data.thinking.finalDecision.strength * 100).toFixed(1)}%
                    </p>
                  </div>
                </motion.div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Whale Alert Overlay - Big Red Pulsing Screen */}
      <AnimatePresence>
        {showWhaleAlert && data.whale && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center whale-alert-overlay"
            onClick={() => { setShowWhaleAlert(false); setWhaleAlertDismissed(true); }}
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              className={`relative max-w-2xl w-full mx-4 p-8 rounded-2xl ${
                data.whale.netFlow > 0 ? 'bg-green-900/95' : 'bg-red-900/95'
              } border-2 ${
                data.whale.netFlow > 0 ? 'border-green-500' : 'border-red-500'
              } shadow-2xl`}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Close button */}
              <button
                onClick={() => { setShowWhaleAlert(false); setWhaleAlertDismissed(true); }}
                className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20"
              >
                <X className="w-5 h-5" />
              </button>

              {/* Alert header */}
              <div className="text-center mb-6">
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  className="text-6xl mb-4"
                >
                  🐋
                </motion.div>
                <h2 className={`text-3xl font-bold ${
                  data.whale.netFlow > 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  WHALE ALERT!
                </h2>
                <p className="text-xl text-white/80 mt-2">
                  {data.whale.netFlow > 0 ? 'Large Withdrawals Detected' : 'Large Deposits Detected'}
                </p>
              </div>

              {/* Alert details */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-black/30 rounded-xl p-4 text-center">
                  <p className="text-gray-400 text-sm">Net Flow</p>
                  <p className={`text-3xl font-bold ${
                    data.whale.netFlow > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {data.whale.netFlow > 0 ? '+' : ''}{data.whale.netFlow.toFixed(1)} BTC
                  </p>
                  <p className="text-gray-400 text-sm">
                    ${Math.abs(data.whale.netFlowUsd).toLocaleString()}
                  </p>
                </div>
                <div className="bg-black/30 rounded-xl p-4 text-center">
                  <p className="text-gray-400 text-sm">Sentiment</p>
                  <p className={`text-3xl font-bold ${
                    data.whale.sentiment === 'BULLISH' ? 'text-green-400' : 
                    data.whale.sentiment === 'BEARISH' ? 'text-red-400' : 'text-yellow-400'
                  }`}>
                    {data.whale.sentiment}
                  </p>
                  <p className="text-gray-400 text-sm">
                    Score: {data.whale.score > 0 ? '+' : ''}{data.whale.score.toFixed(2)}
                  </p>
                </div>
              </div>

              {/* Transaction breakdown */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="flex items-center gap-3 bg-green-500/20 rounded-lg p-3">
                  <ArrowUpRight className="w-6 h-6 text-green-400" />
                  <div>
                    <p className="text-green-400 font-bold">{data.whale.withdrawals} Withdrawals</p>
                    <p className="text-gray-300">{data.whale.withdrawalBtc.toFixed(1)} BTC</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 bg-red-500/20 rounded-lg p-3">
                  <ArrowDownRight className="w-6 h-6 text-red-400" />
                  <div>
                    <p className="text-red-400 font-bold">{data.whale.deposits} Deposits</p>
                    <p className="text-gray-300">{data.whale.depositBtc.toFixed(1)} BTC</p>
                  </div>
                </div>
              </div>

              {/* Analysis */}
              {data.whale.reasoning && (
                <div className="bg-black/30 rounded-lg p-4 mb-4">
                  <p className="text-gray-400 text-sm mb-1">Analysis</p>
                  <p className="text-white">{data.whale.reasoning}</p>
                </div>
              )}

              {/* Large transactions */}
              {data.whale.largeTxs && data.whale.largeTxs.length > 0 && (
                <div className="bg-black/30 rounded-lg p-4">
                  <p className="text-gray-400 text-sm mb-2">Recent Large Transactions</p>
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    {data.whale.largeTxs.slice(0, 5).map((tx, i) => (
                      <div key={i} className="flex items-center justify-between text-sm">
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          tx.type === 'WITHDRAWAL' ? 'bg-green-500/30 text-green-400' :
                          tx.type === 'DEPOSIT' ? 'bg-red-500/30 text-red-400' :
                          'bg-gray-500/30 text-gray-400'
                        }`}>
                          {tx.type}
                        </span>
                        <span className="text-white font-mono">{tx.btc.toFixed(2)} BTC</span>
                        <span className="text-gray-400">${tx.usd_approx?.toLocaleString() || '0'}</span>
                        <span className="text-gray-500 text-xs">{tx.time}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Whale History Panel */}
      <AnimatePresence>
        {showWhaleHistory && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-end bg-black/50"
            onClick={() => setShowWhaleHistory(false)}
          >
            <motion.div
              initial={{ x: 400 }}
              animate={{ x: 0 }}
              exit={{ x: 400 }}
              className="h-full w-full max-w-md bg-[#0a0a0f] border-l border-gray-800 overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="sticky top-0 bg-[#0a0a0f] border-b border-gray-800 p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Fish className="w-6 h-6 text-blue-400" />
                  <h2 className="text-xl font-bold">🐋 Whale Activity</h2>
                </div>
                <button
                  onClick={() => setShowWhaleHistory(false)}
                  className="p-2 rounded-full hover:bg-gray-800"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Current Status */}
              <div className="p-4 border-b border-gray-800">
                <h3 className="text-sm text-gray-400 mb-3">Current Status</h3>
                <div className={`p-4 rounded-xl ${
                  data.whale?.sentiment === 'BULLISH' ? 'bg-green-900/30 border border-green-500/30' :
                  data.whale?.sentiment === 'BEARISH' ? 'bg-red-900/30 border border-red-500/30' :
                  'bg-gray-800/50 border border-gray-700'
                }`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className={`text-2xl font-bold ${
                      data.whale?.sentiment === 'BULLISH' ? 'text-green-400' :
                      data.whale?.sentiment === 'BEARISH' ? 'text-red-400' :
                      'text-gray-400'
                    }`}>
                      {data.whale?.sentiment || 'NEUTRAL'}
                    </span>
                    <span className="text-gray-400">
                      Score: {(data.whale?.score || 0) > 0 ? '+' : ''}{(data.whale?.score || 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-gray-500">Withdrawals</p>
                      <p className="text-green-400 font-mono">{data.whale?.withdrawalBtc?.toFixed(1) || 0} BTC</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Deposits</p>
                      <p className="text-red-400 font-mono">{data.whale?.depositBtc?.toFixed(1) || 0} BTC</p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-gray-500">Net Flow</p>
                      <p className={`font-mono text-lg ${
                        (data.whale?.netFlow || 0) > 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {(data.whale?.netFlow || 0) > 0 ? '+' : ''}{(data.whale?.netFlow || 0).toFixed(1)} BTC
                        <span className="text-gray-500 text-sm ml-2">
                          (${Math.abs(data.whale?.netFlowUsd || 0).toLocaleString()})
                        </span>
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* History */}
              <div className="p-4">
                <h3 className="text-sm text-gray-400 mb-3">History</h3>
                <div className="space-y-3">
                  {data.whaleHistory && data.whaleHistory.length > 0 ? (
                    data.whaleHistory.map((entry, i) => (
                      <div key={i} className={`p-3 rounded-lg ${
                        entry.sentiment === 'BULLISH' ? 'bg-green-900/20 border-l-2 border-green-500' :
                        entry.sentiment === 'BEARISH' ? 'bg-red-900/20 border-l-2 border-red-500' :
                        'bg-gray-800/30 border-l-2 border-gray-600'
                      }`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className={`font-medium ${
                            entry.sentiment === 'BULLISH' ? 'text-green-400' :
                            entry.sentiment === 'BEARISH' ? 'text-red-400' :
                            'text-gray-400'
                          }`}>
                            {entry.sentiment}
                          </span>
                          <span className="text-xs text-gray-500">
                            {entry.timestamp} {entry.date !== new Date().toISOString().split('T')[0] && entry.date}
                          </span>
                        </div>
                        <div className="text-sm">
                          <span className={`font-mono ${
                            entry.netFlow > 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {entry.netFlow > 0 ? '+' : ''}{entry.netFlow.toFixed(1)} BTC
                          </span>
                          <span className="text-gray-500 ml-2">net flow</span>
                        </div>
                        {entry.reasoning && (
                          <p className="text-xs text-gray-500 mt-1 truncate">{entry.reasoning}</p>
                        )}
                      </div>
                    ))
                  ) : (
                    <p className="text-gray-500 text-center py-8">No whale activity recorded yet</p>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Wisdom Oracle Detail Modal */}
      <AnimatePresence>
        {showWisdomDetail && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={() => setShowWisdomDetail(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto bg-[#0a0a0f] border border-[#00d4aa]/30 rounded-2xl shadow-2xl flex"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Left side - Trader Image */}
              {(data.signals.wisdom.master?.toLowerCase().includes('buffett') || data.signals.wisdom.master?.toLowerCase().includes('soros')) && (
                <div className="flex w-48 md:w-64 flex-shrink-0 bg-gradient-to-b from-[#00d4aa]/10 to-purple-600/10 items-center justify-center p-4 rounded-l-2xl">
                  <motion.div 
                    className="w-full rounded-xl overflow-hidden border-2 border-[#00d4aa]/50 shadow-xl"
                    animate={{ scale: [1, 1.02, 1] }}
                    transition={{ duration: 3, repeat: Infinity }}
                  >
                    <img 
                      src={data.signals.wisdom.master?.toLowerCase().includes('buffett') ? '/warren-buffett.jpg' : '/george-soros.jpg'}
                      alt={data.signals.wisdom.master || 'Trader'} 
                      className="w-full h-auto object-cover"
                    />
                  </motion.div>
                </div>
              )}
              
              {/* Right side - Content */}
              <div className="flex-1">
                {/* Header */}
                <div className="sticky top-0 bg-gradient-to-r from-[#00d4aa]/20 to-purple-600/20 border-b border-gray-800 p-6">
                  <button
                    onClick={() => setShowWisdomDetail(false)}
                    className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 z-10"
                  >
                    <X className="w-5 h-5" />
                  </button>
                  
                  <div className="flex items-center gap-4">
                    {/* Small avatar for mobile or non-image traders */}
                    <motion.div 
                      className="w-16 h-16 md:w-20 md:h-20 rounded-full bg-gradient-to-br from-[#00d4aa] to-purple-600 flex items-center justify-center text-3xl md:text-4xl flex-shrink-0"
                      animate={{ rotate: [0, 5, -5, 0] }}
                      transition={{ duration: 4, repeat: Infinity }}
                    >
                      {data.signals.wisdom.master?.includes('Livermore') ? '📈' :
                       data.signals.wisdom.master?.includes('Soros') ? '🌍' :
                       data.signals.wisdom.master?.includes('Jones') ? '📊' :
                       data.signals.wisdom.master?.includes('Buffett') ? '🦉' :
                       data.signals.wisdom.master?.includes('Dalio') ? '⚙️' : '🧙'}
                    </motion.div>
                    <div className="min-w-0 flex-1">
                      <h2 className="text-xl md:text-2xl font-bold text-white truncate">
                        {data.signals.wisdom.master || 'Wisdom Oracle'}
                      </h2>
                      <p className="text-[#00d4aa] text-sm">{data.signals.wisdom.specialty || 'Market Analysis'}</p>
                      <div className="flex items-center gap-2 mt-1 flex-wrap">
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          data.signals.wisdom.style === 'aggressive' ? 'bg-red-500/20 text-red-400' :
                          data.signals.wisdom.style === 'contrarian' ? 'bg-purple-500/20 text-purple-400' :
                          data.signals.wisdom.style === 'technical' ? 'bg-blue-500/20 text-blue-400' :
                          data.signals.wisdom.style === 'patient' ? 'bg-green-500/20 text-green-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {data.signals.wisdom.style?.toUpperCase() || 'SYSTEMATIC'}
                        </span>
                        <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                          data.signals.wisdom.grade.includes('A') ? 'bg-green-500/30 text-green-400' : 
                          data.signals.wisdom.grade === 'B' ? 'bg-yellow-500/30 text-yellow-400' : 'bg-gray-500/30 text-gray-400'
                        }`}>
                          Grade: {data.signals.wisdom.grade}
                        </span>
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0">
                      <div className={`text-2xl md:text-4xl font-bold ${
                        data.signals.wisdom.signal.includes('BUY') ? 'text-green-400' :
                        data.signals.wisdom.signal.includes('SELL') ? 'text-red-400' :
                        'text-yellow-400'
                      }`}>
                        {data.signals.wisdom.signal}
                      </div>
                    </div>
                  </div>
                </div>

              {/* Content */}
              <div className="p-6 space-y-6">
                {/* Selection Reason */}
                <div className="bg-[#00d4aa]/10 border border-[#00d4aa]/30 rounded-xl p-4">
                  <h3 className="text-sm text-[#00d4aa] font-medium mb-2">🎭 Why This Trader Was Selected</h3>
                  <p className="text-white">{data.signals.wisdom.selectionReason || 'Market conditions analyzed...'}</p>
                </div>

                {/* Key Levels */}
                {data.signals.wisdom.keyLevels && Object.keys(data.signals.wisdom.keyLevels).length > 0 && (
                  <div className="bg-gray-800/50 rounded-xl p-4">
                    <h3 className="text-sm text-gray-400 font-medium mb-3">📊 Key Price Levels</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-green-500/10 rounded-lg p-3">
                        <p className="text-xs text-gray-500">Support</p>
                        <p className="text-lg font-bold text-green-400">
                          ${data.signals.wisdom.keyLevels.support?.toLocaleString() || 'N/A'}
                        </p>
                      </div>
                      <div className="bg-red-500/10 rounded-lg p-3">
                        <p className="text-xs text-gray-500">Resistance</p>
                        <p className="text-lg font-bold text-red-400">
                          ${data.signals.wisdom.keyLevels.resistance?.toLocaleString() || 'N/A'}
                        </p>
                      </div>
                      <div className="bg-red-500/10 rounded-lg p-3">
                        <p className="text-xs text-gray-500">Stop Loss</p>
                        <p className="text-lg font-bold text-red-400">
                          ${data.signals.wisdom.keyLevels.stop_loss?.toLocaleString() || 'N/A'}
                        </p>
                      </div>
                      <div className="bg-green-500/10 rounded-lg p-3">
                        <p className="text-xs text-gray-500">Take Profit</p>
                        <p className="text-lg font-bold text-green-400">
                          ${data.signals.wisdom.keyLevels.take_profit?.toLocaleString() || 'N/A'}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Full Reasoning */}
                <div className="bg-gray-800/50 rounded-xl p-4">
                  <h3 className="text-sm text-gray-400 font-medium mb-3">💭 Oracle's Analysis</h3>
                  <p className="text-white leading-relaxed">
                    {data.signals.wisdom.reasoning || 'Awaiting oracle response...'}
                  </p>
                </div>

                {/* Warnings */}
                {data.signals.wisdom.fullAnalysis?.warnings && data.signals.wisdom.fullAnalysis.warnings.length > 0 && (
                  <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
                    <h3 className="text-sm text-yellow-400 font-medium mb-2">⚠️ Warnings</h3>
                    <ul className="space-y-1">
                      {data.signals.wisdom.fullAnalysis.warnings.map((warning: string, i: number) => (
                        <li key={i} className="text-yellow-200 text-sm flex items-start gap-2">
                          <span className="text-yellow-500">•</span>
                          {warning}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Full Analysis Details */}
                {data.signals.wisdom.fullAnalysis && (
                  <div className="bg-gray-800/30 rounded-xl p-4">
                    <h3 className="text-sm text-gray-400 font-medium mb-3">📋 Full Analysis Data</h3>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Signal Score:</span>
                        <span className={`font-mono ${
                          (data.signals.wisdom.fullAnalysis.signal_score || 0) > 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {(data.signals.wisdom.fullAnalysis.signal_score || 0) > 0 ? '+' : ''}
                          {(data.signals.wisdom.fullAnalysis.signal_score || 0).toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Confidence:</span>
                        <span className="text-white font-mono">
                          {((data.signals.wisdom.fullAnalysis.confidence || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Risk/Reward:</span>
                        <span className="text-white font-mono">
                          {(data.signals.wisdom.fullAnalysis.risk_reward_ratio || 0).toFixed(1)}:1
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Trade Quality:</span>
                        <span className={`font-bold ${
                          data.signals.wisdom.fullAnalysis.trade_quality?.includes('A') ? 'text-green-400' :
                          data.signals.wisdom.fullAnalysis.trade_quality === 'B' ? 'text-yellow-400' :
                          'text-gray-400'
                        }`}>
                          {data.signals.wisdom.fullAnalysis.trade_quality || 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quant Analysis Detail Modal */}
      <AnimatePresence>
        {showQuantDetail && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={() => setShowQuantDetail(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto bg-[#0a0a0f] border border-yellow-500/30 rounded-2xl shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="sticky top-0 bg-gradient-to-r from-yellow-500/20 to-orange-600/20 border-b border-gray-800 p-6">
                <button
                  onClick={() => setShowQuantDetail(false)}
                  className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 z-10"
                >
                  <X className="w-5 h-5" />
                </button>
                
                <div className="flex items-center gap-4">
                  <motion.div 
                    className="w-16 h-16 rounded-full bg-gradient-to-br from-yellow-500 to-orange-600 flex items-center justify-center text-3xl"
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                  >
                    📐
                  </motion.div>
                  <div>
                    <h2 className="text-2xl font-bold text-white">Quant Analysis</h2>
                    <p className="text-yellow-400 text-sm">Statistical & Risk Metrics</p>
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="p-6 space-y-4">
                {/* Signal */}
                <div className="bg-gray-800/50 rounded-xl p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Signal</span>
                    <span className={`text-2xl font-bold ${getSignalColor(data.signals.quant.signal)}`}>
                      {data.signals.quant.signal}
                    </span>
                  </div>
                </div>

                {/* Z-Score & Momentum */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-800/50 rounded-xl p-4">
                    <span className="text-gray-400 text-sm block mb-2">Z-Score</span>
                    <span className={`text-3xl font-bold ${data.signals.quant.zScore >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {data.signals.quant.zScore >= 0 ? '+' : ''}{data.signals.quant.zScore.toFixed(2)}
                    </span>
                    <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden relative">
                      <div className="absolute left-1/2 w-0.5 h-full bg-gray-500" />
                      <div 
                        className={`h-full ${data.signals.quant.zScore >= 0 ? 'bg-green-500' : 'bg-red-500'}`}
                        style={{ 
                          width: `${Math.min(Math.abs(data.signals.quant.zScore) * 16, 50)}%`,
                          marginLeft: data.signals.quant.zScore >= 0 ? '50%' : `${50 - Math.min(Math.abs(data.signals.quant.zScore) * 16, 50)}%`
                        }}
                      />
                    </div>
                    <div className="text-xs text-gray-500 mt-2">
                      {data.signals.quant.zScore > 2 ? '⚠️ Overbought' : data.signals.quant.zScore < -2 ? '⚠️ Oversold' : 'Normal range'}
                    </div>
                  </div>

                  <div className="bg-gray-800/50 rounded-xl p-4">
                    <span className="text-gray-400 text-sm block mb-2">Volatility</span>
                    <span className={`text-2xl font-bold ${
                      data.signals.quant.volatilityRegime === 'EXTREME' ? 'text-red-400' :
                      data.signals.quant.volatilityRegime === 'HIGH' ? 'text-orange-400' :
                      data.signals.quant.volatilityRegime === 'LOW' ? 'text-green-400' : 'text-yellow-400'
                    }`}>
                      {data.signals.quant.volatilityRegime || 'NORMAL'}
                    </span>
                    <div className="text-sm text-gray-400 mt-1">
                      {(data.signals.quant.volatilityPctl || 50).toFixed(0)}th percentile
                    </div>
                    <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${
                          (data.signals.quant.volatilityPctl || 50) > 80 ? 'bg-red-500' :
                          (data.signals.quant.volatilityPctl || 50) > 60 ? 'bg-orange-500' :
                          (data.signals.quant.volatilityPctl || 50) < 30 ? 'bg-green-500' : 'bg-yellow-500'
                        }`}
                        style={{ width: `${data.signals.quant.volatilityPctl || 50}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Momentum */}
                <div className="bg-gray-800/50 rounded-xl p-4">
                  <span className="text-gray-400 text-sm block mb-3">Momentum</span>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-xs text-gray-500 mb-1">5 Period</div>
                      <div className={`text-lg font-bold ${(data.signals.quant.momentum5p || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(data.signals.quant.momentum5p || 0) >= 0 ? '+' : ''}{(data.signals.quant.momentum5p || 0).toFixed(2)}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-xs text-gray-500 mb-1">10 Period</div>
                      <div className={`text-lg font-bold ${(data.signals.quant.momentum10p || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(data.signals.quant.momentum10p || 0) >= 0 ? '+' : ''}{(data.signals.quant.momentum10p || 0).toFixed(2)}%
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-xs text-gray-500 mb-1">20 Period</div>
                      <div className={`text-lg font-bold ${(data.signals.quant.momentum20p || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(data.signals.quant.momentum20p || 0) >= 0 ? '+' : ''}{(data.signals.quant.momentum20p || 0).toFixed(2)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* HTF Trend */}
                <div className="bg-gray-800/50 rounded-xl p-4">
                  <span className="text-gray-400 text-sm block mb-2">Higher Timeframe Trend</span>
                  <div className="flex items-center justify-between">
                    <span className={`text-2xl font-bold ${
                      data.signals.quant.htfTrend === 'BULLISH' ? 'text-green-400' :
                      data.signals.quant.htfTrend === 'BEARISH' ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {data.signals.quant.htfTrend || 'NEUTRAL'}
                    </span>
                    <span className={`text-lg ${(data.signals.quant.htfBias || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      Bias: {(data.signals.quant.htfBias || 0) >= 0 ? '+' : ''}{(data.signals.quant.htfBias || 0).toFixed(2)}
                    </span>
                  </div>
                </div>

                {/* Kelly & Position Sizing */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-xl p-4 border border-cyan-500/30">
                    <span className="text-gray-400 text-sm block mb-2">Kelly Criterion</span>
                    <span className="text-3xl font-bold text-cyan-400">
                      {((data.signals.quant.kellyFraction || 0) * 100).toFixed(1)}%
                    </span>
                    <div className="text-xs text-gray-400 mt-1">Optimal position size</div>
                  </div>
                  <div className="bg-gradient-to-br from-green-500/20 to-emerald-600/20 rounded-xl p-4 border border-green-500/30">
                    <span className="text-gray-400 text-sm block mb-2">Suggested Size</span>
                    <span className="text-3xl font-bold text-green-400">
                      ${(data.signals.quant.optimalSize || 0).toFixed(2)}
                    </span>
                    <div className="text-xs text-gray-400 mt-1">Based on Kelly & volatility</div>
                  </div>
                </div>

                {/* Reasoning */}
                {data.signals.quant.reasoning && (
                  <div className="bg-gray-800/50 rounded-xl p-4">
                    <span className="text-gray-400 text-sm block mb-2">Analysis</span>
                    <p className="text-white">{data.signals.quant.reasoning}</p>
                  </div>
                )}

                {/* Risk Metrics from Learning */}
                {(data.learning?.totalTrades || 0) >= 5 && (
                  <div className="bg-gradient-to-br from-purple-500/10 to-pink-600/10 rounded-xl p-4 border border-purple-500/30">
                    <span className="text-gray-400 text-sm block mb-3">📊 Performance Metrics (Last 30 Days)</span>
                    <div className="grid grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Expectancy</div>
                        <div className={`text-lg font-bold ${(data.learning?.expectancy || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {(data.learning?.expectancy || 0).toFixed(2)}%
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Max DD</div>
                        <div className="text-lg font-bold text-orange-400">
                          {(data.learning?.maxDrawdown || 0).toFixed(1)}%
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Sharpe</div>
                        <div className={`text-lg font-bold ${(data.learning?.sharpeRatio || 0) >= 1 ? 'text-green-400' : 'text-yellow-400'}`}>
                          {(data.learning?.sharpeRatio || 0).toFixed(1)}
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Profit Factor</div>
                        <div className={`text-lg font-bold ${(data.learning?.profitFactor || 0) >= 1.5 ? 'text-green-400' : 'text-yellow-400'}`}>
                          {(data.learning?.profitFactor || 0).toFixed(1)}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
