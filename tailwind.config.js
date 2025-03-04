/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./src/frontend/templates/**/*.html", "./src/frontend/static/js/**/*.js"],
    plugins: [
        require("@tailwindcss/typography"),
    ],
} 