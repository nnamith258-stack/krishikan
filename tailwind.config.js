/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      boxShadow: {
        soft: '0 18px 40px rgba(15, 118, 110, 0.12)',
      },
      colors: {
        ember: '#f59e0b',
        danger: '#dc2626',
        forest: '#14532d',
        olive: '#4d7c0f',
        leaf: '#22c55e',
        sky: '#2563eb',
      },
    },
  },
  plugins: [],
}

