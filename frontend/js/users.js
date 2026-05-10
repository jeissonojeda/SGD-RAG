// ═══════════════════════════════════════════════
// users.js — Gestión de usuarios
// ═══════════════════════════════════════════════

async function loadUsers() {
  try {
    const res = await fetch(`${API}/api/users`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) return;
    const users = await res.json();
    const grid = document.getElementById('users-grid');
    grid.innerHTML = users.map(u => `
      <div class="user-card">
        <div class="user-card-top">
          <div class="user-card-avatar">${u.username[0].toUpperCase()}</div>
          <div>
            <div class="user-card-name">${u.username}</div>
            <div class="user-card-email">${u.email}</div>
          </div>
        </div>
        <span class="role-badge ${u.role === 'Administrador' ? 'admin' : ''}">${u.role}</span>
      </div>
    `).join('');
  } catch (e) { console.error(e); }
}

async function addUser() {
  const username = document.getElementById('nu-user').value.trim();
  const email    = document.getElementById('nu-email').value.trim();
  const password = document.getElementById('nu-pass').value;
  const role     = document.getElementById('nu-role').value;

  if (!username || !email || !password) return toast('Completa todos los campos', 'error');

  try {
    const res = await fetch(`${API}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ username, email, password, role })
    });
    const data = await res.json();
    if (res.ok) {
      toast('Usuario creado exitosamente');
      loadUsers();
      // Limpiar campos
      document.getElementById('nu-user').value = '';
      document.getElementById('nu-email').value = '';
      document.getElementById('nu-pass').value = '';
    } else {
      toast(data.detail || 'Error al crear usuario', 'error');
    }
  } catch (e) { toast('Error de conexión', 'error'); }
}
