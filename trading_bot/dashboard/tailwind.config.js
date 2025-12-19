/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
    './app/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'brand': {
          dark: '#0a0a0f',
          darker: '#050508',
          accent: '#00d4aa',
          red: '#ff4757',
          green: '#2ed573',
          orange: '#ffa502',
          blue: '#3742fa',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px #00d4aa, 0 0 10px #00d4aa' },
          '100%': { boxShadow: '0 0 10px #00d4aa, 0 0 20px #00d4aa, 0 0 30px #00d4aa' },
        }
      }
    },
  },
  plugins: [],
}
