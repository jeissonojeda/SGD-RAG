const API = "http://127.0.0.1:8000";

// ───────── LOGIN ─────────
async function doLogin() {
  const username = document.getElementById("login-user").value;
  const password = document.getElementById("login-pass").value;

  if (!username || !password) {
    alert("Ingrese usuario y contraseña");
    return;
  }

  try {
    const formData = new URLSearchParams();
    formData.append("username", username);
    formData.append("password", password);

    const res = await fetch(`${API}/api/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      alert(data.detail || "Error al iniciar sesión");
      return;
    }

    // Guardar sesión
    localStorage.setItem("sgd_token", data.access_token);
    localStorage.setItem("sgd_user", JSON.stringify(data.user));

    // Mostrar bienvenida
    showWelcomeScreen();

  } catch (err) {
    console.error(err);
    alert("Error de conexión con el servidor");
  }
}


// ───────── REGISTRO ─────────
async function doRegister() {
  const username = document.getElementById("reg-user").value;
  const email = document.getElementById("reg-email").value;
  const password = document.getElementById("reg-pass").value;

  if (!username || !email || !password) {
    alert("Complete todos los campos");
    return;
  }

  try {
    const res = await fetch(`${API}/api/auth/register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ username, email, password })
    });

    const data = await res.json();

    if (!res.ok) {
      alert(data.detail || "Error al registrar");
      return;
    }

    alert("Usuario creado correctamente");

  } catch (err) {
    console.error(err);
    alert("Error de conexión");
  }
}


// ───────── VALIDAR TOKEN ─────────
async function checkAuth() {
  const token = localStorage.getItem("sgd_token");

  if (!token) return false;

  try {
    const res = await fetch(`${API}/api/auth/me`, {
      headers: {
        "Authorization": "Bearer " + token
      }
    });

    if (!res.ok) {
      return false;
    }

    const user = await res.json();
    localStorage.setItem("sgd_user", JSON.stringify(user));

    return true;

  } catch {
    return false;
  }
}


// ───────── ENTRAR A LA APP ─────────
function enterApp() {
  document.getElementById("login-screen").style.display = "none";
  document.getElementById("app").style.display = "block";
}


// ───────── LOGOUT ─────────
function doLogout() {
  localStorage.removeItem("sgd_token");
  localStorage.removeItem("sgd_user");
  location.reload();
}


// ───────── BIENVENIDA ─────────
function showWelcomeScreen() {
  const welcome = document.getElementById("welcome-screen");

  if (!welcome) {
    enterApp();
    return;
  }

  const user = JSON.parse(localStorage.getItem("sgd_user") || "{}");

  document.getElementById("welcome-username").textContent =
    user.username || "Usuario";

  document.getElementById("welcome-role").textContent =
    user.role || "Usuario";

  welcome.style.display = "block";

  setTimeout(() => {
    welcome.classList.add("hide");
    enterApp();
  }, 4000);
}


// ───────── INICIO AUTOMÁTICO ─────────
window.addEventListener("load", async () => {

  const token = localStorage.getItem("sgd_token");

  // 🔴 SIN TOKEN → LOGIN DIRECTO
  if (!token) {
    document.getElementById("welcome-screen")?.classList.add("hide");
    document.getElementById("login-screen").style.display = "block";
    return;
  }

  // 🔵 CON TOKEN → VALIDAR
  const isAuth = await checkAuth();

  if (isAuth) {
    showWelcomeScreen();
  } else {
    doLogout();
  }

});