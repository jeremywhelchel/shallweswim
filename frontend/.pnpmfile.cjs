// openapi-typescript builds the generated API client with the TypeScript 5
// compiler API, which TypeScript 7 no longer ships. Give it its own
// TypeScript 5 so the project's tsc can be TypeScript 7.
function readPackage(pkg) {
  if (pkg.name === "openapi-typescript") {
    delete pkg.peerDependencies?.typescript;
    pkg.dependencies = { ...pkg.dependencies, typescript: "5.9.3" };
  }
  return pkg;
}

module.exports = { hooks: { readPackage } };
