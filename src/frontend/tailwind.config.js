/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'az-blue':    '#0033A0',
        'az-gold':    '#EFAB00',
        'az-magenta': '#8A0051',
        'az-dark':    '#0A0F1E',
        'az-card':    '#0D1526',
        'az-border':  '#1E2D4A',
        'az-muted':   '#6B7280',
      },
    },
  },
  plugins: [],
}
