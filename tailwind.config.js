/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          50: '#f6f8fa',
          100: '#eef1f5',
          200: '#d0d7de',
          300: '#afb8c1',
          400: '#8b949e',
          500: '#656d76',
          600: '#484f58',
          700: '#21262d',
          800: '#161b22',
          850: '#0e1117',
          900: '#07080a',
        },
        accent: {
          DEFAULT: '#c9985a',
          light: '#d4a864',
          dark: '#9a7035',
          50: '#fdf8f0',
          100: '#f5e6cd',
          200: '#e8cfa0',
          300: '#d4a864',
          400: '#c9985a',
          500: '#b5863f',
          600: '#9a7035',
          700: '#7a582a',
        },
        up: '#3fb950',
        down: '#f85149',
        info: '#58a6ff',
        warn: '#d29922',
      },
      fontFamily: {
        display: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.65rem', { lineHeight: '1rem' }],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'slide-right': 'slideRight 0.3s ease-out',
        'pulse-soft': 'pulseSoft 2s infinite',
        'number-tick': 'numberTick 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideRight: {
          '0%': { opacity: '0', transform: 'translateX(-12px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
        numberTick: {
          '0%': { transform: 'translateY(-4px)', opacity: '0.5' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      backgroundImage: {
        'grain': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E\")",
      },
    },
  },
  plugins: [],
}
